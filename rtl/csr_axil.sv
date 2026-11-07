// csr_axil.sv -- control/status registers over AXI4-Lite (32-bit data, 12-bit address).
// Register map: docs/control.md §5 (kept in sync with tests/rtl/ctrl_cocotb.py::REG).
//
// AW and W are accepted independently (either order, or together); the write happens when both
// are held, then B is returned. One read in flight: AR is accepted while R is not pending.
// Writes to read-only or unmapped addresses are ignored (OKAY); unmapped reads return 0.
module csr_axil #(
    parameter int PW   = 24,
    parameter int NSAT = 4
) (
    input  logic                 clk,
    input  logic                 rst_n,
    // AXI4-Lite slave
    input  logic [11:0]          s_axil_awaddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [2:0]           s_axil_awprot,
    input  logic [2:0]           s_axil_arprot,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                 s_axil_awvalid,
    output logic                 s_axil_awready,
    input  logic [31:0]          s_axil_wdata,
    input  logic [3:0]           s_axil_wstrb,
    input  logic                 s_axil_wvalid,
    output logic                 s_axil_wready,
    output logic [1:0]           s_axil_bresp,
    output logic                 s_axil_bvalid,
    input  logic                 s_axil_bready,
    input  logic [11:0]          s_axil_araddr,
    input  logic                 s_axil_arvalid,
    output logic                 s_axil_arready,
    output logic [31:0]          s_axil_rdata,
    output logic [1:0]           s_axil_rresp,
    output logic                 s_axil_rvalid,
    input  logic                 s_axil_rready,
    // configuration out
    output logic [31:0]          ring_base,
    output logic [PW-1:0]        ring_size,
    output logic [PW-1:0]        ring_tail,
    output logic                 doorbell,       // pulse: tail written
    output logic                 step_mode,
    output logic                 step_pulse,
    output logic                 soft_req,       // pulse: soft reset requested
    output logic [4:0]           sem_sel,
    output logic                 sat_clr,        // pulse: clear saturation counters
    output logic                 irq,
    // status in
    input  logic                 clr,            // soft reset in progress -> clear run state
    input  logic                 soft_busy,
    input  logic [PW-1:0]        ring_head,
    input  logic                 cfg_ok,
    input  logic                 halted,
    input  logic [3:0]           err_code,
    input  logic [PW-1:0]        err_index,
    input  logic [127:0]         err_word,
    input  logic                 irq_pulse,
    input  logic [15:0]          irq_code,
    input  logic                 end_pulse,
    input  logic                 all_idle,
    input  logic [2:0]           q_idle,
    input  logic [2:0]           eng_busy,
    input  logic [2:0]           retire_inc,
    input  logic [PW-1:0]        last_index,
    input  logic                 q_overflow,
    input  logic [7:0]           sem_val,
    input  logic                 dma_bytes_valid,
    input  logic [16:0]          dma_bytes,
    input  logic [NSAT*32-1:0]   sat_cnt
);
    localparam logic [31:0] ID_VALUE = 32'h4E58_4C43;   // "NXLC"
    localparam logic [31:0] VERSION  = 32'h0001_0000;   // control plane v1.0

    localparam logic [11:0]
        A_ID = 12'h000, A_VERSION = 12'h004, A_CTRL = 12'h008, A_STATUS = 12'h00C,
        A_ERR_CODE = 12'h010, A_ERR_INDEX = 12'h014, A_ERR_W0 = 12'h018,
        A_RING_BASE = 12'h030, A_RING_SIZE = 12'h034, A_RING_HEAD = 12'h038, A_DOORBELL = 12'h03C,
        A_IRQ_EN = 12'h040, A_IRQ_STATUS = 12'h044, A_IRQ_CODE = 12'h048, A_END_COUNT = 12'h04C,
        A_DBG_STEP = 12'h050, A_DBG_LAST = 12'h054, A_SEM_SEL = 12'h058, A_SEM_VAL = 12'h05C,
        A_PERF_CTRL = 12'h060, A_PERF_CYCLES = 12'h064, A_PERF_BUSY0 = 12'h068,
        A_PERF_RETIRED = 12'h074, A_PERF_BYTES_LO = 12'h078, A_PERF_BYTES_HI = 12'h07C,
        A_SAT0 = 12'h080;

    // ---------------- AXI4-Lite write ----------------
    logic        aw_held, w_held;
    logic [11:0] aw_addr;
    logic [31:0] w_data;
    logic [3:0]  w_strb;
    logic        wr_fire;

    assign s_axil_awready = !aw_held;
    assign s_axil_wready  = !w_held;
    assign s_axil_bresp   = 2'b00;
    assign wr_fire        = aw_held && w_held && !s_axil_bvalid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            aw_held       <= 1'b0;
            w_held        <= 1'b0;
            s_axil_bvalid <= 1'b0;
        end else begin
            if (s_axil_awvalid && s_axil_awready) begin aw_held <= 1'b1; aw_addr <= s_axil_awaddr; end
            if (s_axil_wvalid && s_axil_wready)   begin w_held <= 1'b1; w_data <= s_axil_wdata; w_strb <= s_axil_wstrb; end
            if (wr_fire) begin
                aw_held       <= 1'b0;
                w_held        <= 1'b0;
                s_axil_bvalid <= 1'b1;
            end
            if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
        end
    end

    // byte-strobed merge into an existing register value
    function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] d, input logic [3:0] s);
        for (int b = 0; b < 4; b++) merge[8*b +: 8] = s[b] ? d[8*b +: 8] : old[8*b +: 8];
    endfunction

    // ---------------- registers ----------------
    logic [2:0]  irq_en, irq_status;
    logic [31:0] end_count, perf_cycles, perf_retired;
    logic [31:0] perf_busy [0:2];
    logic [63:0] perf_bytes;
    logic        halted_q;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [31:0] wv;
    /* verilator lint_on UNUSEDSIGNAL */

    assign wv = merge(32'h0, w_data, w_strb);

    always_ff @(posedge clk) begin
        doorbell   <= 1'b0;
        step_pulse <= 1'b0;
        soft_req   <= 1'b0;
        sat_clr    <= 1'b0;
        if (!rst_n) begin
            ring_base  <= '0;
            ring_size  <= '0;
            ring_tail  <= '0;
            step_mode  <= 1'b0;
            sem_sel    <= '0;
            irq_en     <= '0;
            irq_status <= '0;
            end_count  <= '0;
            halted_q   <= 1'b0;
            perf_cycles <= '0; perf_retired <= '0; perf_bytes <= '0;
            for (int i = 0; i < 3; i++) perf_busy[i] <= '0;
        end else begin
            halted_q <= halted;
            // events
            if (end_pulse) begin irq_status[0] <= 1'b1; end_count <= end_count + 1'b1; end
            if (irq_pulse) irq_status[1] <= 1'b1;
            if (halted && !halted_q) irq_status[2] <= 1'b1;
            // perf counters
            if (!(all_idle && ring_head == ring_tail) && !halted) perf_cycles <= perf_cycles + 1'b1;
            for (int i = 0; i < 3; i++) if (eng_busy[i]) perf_busy[i] <= perf_busy[i] + 1'b1;
            perf_retired <= perf_retired + 32'(retire_inc);
            if (dma_bytes_valid) perf_bytes <= perf_bytes + 64'(dma_bytes);
            // register writes
            if (wr_fire) begin
                case (aw_addr)
                    A_CTRL: begin
                        if (w_strb[0]) begin
                            step_mode <= w_data[0];
                            if (w_data[1]) soft_req <= 1'b1;
                        end
                    end
                    A_RING_BASE: ring_base <= {merge(ring_base, w_data, w_strb)} & 32'hFFFF_FFF0;
                    A_RING_SIZE: ring_size <= PW'(merge(32'(ring_size), w_data, w_strb));
                    A_DOORBELL:  begin ring_tail <= PW'(merge(32'(ring_tail), w_data, w_strb)); doorbell <= 1'b1; end
                    A_IRQ_EN:    irq_en <= 3'(merge(32'(irq_en), w_data, w_strb));
                    A_IRQ_STATUS: irq_status <= irq_status & ~3'(wv);   // write 1 to clear
                    A_DBG_STEP:  step_pulse <= 1'b1;
                    A_SEM_SEL:   sem_sel <= 5'(merge(32'(sem_sel), w_data, w_strb));
                    A_PERF_CTRL: begin
                        if (wv[0]) begin
                            perf_cycles <= '0; perf_retired <= '0; perf_bytes <= '0;
                            for (int i = 0; i < 3; i++) perf_busy[i] <= '0;
                        end
                        if (wv[1]) sat_clr <= 1'b1;
                    end
                    default: ;
                endcase
            end
            // a soft reset empties the ring (head is cleared by the fetcher)
            if (clr) ring_tail <= '0;
        end
    end

    assign irq = |(irq_status & irq_en);

    // ---------------- AXI4-Lite read ----------------
    logic [31:0] rd;
    always_comb begin
        logic [11:0] a;
        a  = s_axil_araddr;
        rd = 32'h0;
        case (a)
            A_ID:          rd = ID_VALUE;
            A_VERSION:     rd = VERSION;
            A_CTRL:        rd = {30'b0, soft_busy, step_mode};
            A_STATUS: begin
                rd[0]     = halted;
                rd[1]     = all_idle;                       // queues empty, nothing outstanding
                rd[2]     = (ring_head == ring_tail);       // ring empty
                rd[3]     = !(all_idle && ring_head == ring_tail);  // busy
                rd[6:4]   = q_idle;
                rd[10:8]  = eng_busy;
                rd[13]    = cfg_ok;
                rd[14]    = soft_busy;
                rd[15]    = q_overflow;
            end
            A_ERR_CODE:    rd = {28'b0, err_code};
            A_ERR_INDEX:   rd = 32'(err_index);
            A_RING_BASE:   rd = ring_base;
            A_RING_SIZE:   rd = 32'(ring_size);
            A_RING_HEAD:   rd = 32'(ring_head);
            A_DOORBELL:    rd = 32'(ring_tail);
            A_IRQ_EN:      rd = {29'b0, irq_en};
            A_IRQ_STATUS:  rd = {29'b0, irq_status};
            A_IRQ_CODE:    rd = {16'b0, irq_code};
            A_END_COUNT:   rd = end_count;
            A_DBG_LAST:    rd = 32'(last_index);
            A_SEM_SEL:     rd = {27'b0, sem_sel};
            A_SEM_VAL:     rd = {24'b0, sem_val};
            A_PERF_CYCLES: rd = perf_cycles;
            A_PERF_RETIRED: rd = perf_retired;
            A_PERF_BYTES_LO: rd = perf_bytes[31:0];
            A_PERF_BYTES_HI: rd = perf_bytes[63:32];
            default: begin
                if (a >= A_ERR_W0 && a < A_ERR_W0 + 12'd16)
                    rd = err_word[32*((a - A_ERR_W0) >> 2) +: 32];
                else if (a >= A_PERF_BUSY0 && a < A_PERF_BUSY0 + 12'd12)
                    rd = perf_busy[2'((a - A_PERF_BUSY0) >> 2)];
                else if (a >= A_SAT0 && 32'(a) < 32'(A_SAT0) + 4 * NSAT)
                    rd = sat_cnt[32*((a - A_SAT0) >> 2) +: 32];
            end
        endcase
    end

    assign s_axil_arready = !s_axil_rvalid;
    assign s_axil_rresp   = 2'b00;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_axil_rvalid <= 1'b0;
        end else begin
            if (s_axil_arvalid && s_axil_arready) begin
                s_axil_rvalid <= 1'b1;
                s_axil_rdata  <= rd;
            end else if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end
endmodule
