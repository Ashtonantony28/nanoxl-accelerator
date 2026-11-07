// dma_engine.sv -- DMA_LOAD / DMA_STORE engine: AXI4 master (128-bit) <-> scratchpad line port.
//
// Command port: one channel of the DMA queue's engine port (cmd_valid/cmd_ready + the 128-bit ISA
// word). `done` pulses once per accepted command, in acceptance order, when the command has fully
// completed (LOAD: every scratchpad write accepted; STORE: every write response received).
//
// Pipelining (SG10a, ADR-005): accepted commands are decoded into descriptors and pushed into two
// in-order FIFOs of NCMD entries: the address stage (AR/AW issue) pops `aq` as soon as it has
// issued the previous command's last burst, the data stage (realigner + scratchpad writes, or W
// beats) works on the head of `rq` and pops it at `done`. So the next LOAD's ARs go out while
// earlier LOADs' data is still landing; read data stays in order in the load FIFO, and the data
// stage switches commands without draining the AXI side. A STORE is exclusive: it is accepted only
// into an empty engine and nothing is accepted behind it, so a later LOAD sees an earlier STORE's
// DDR data and a STORE sees every earlier LOAD's scratchpad writes (docs/isa.md §4).
//
// Geometry (docs/isa.md, docs/control.md §DMA): row r moves row_bytes bytes between DDR byte
// address ddr_addr + r*ds and scratchpad byte address sp_addr + r*ss of buffer sp_buf, where
// ds = ddr_stride (0 -> row_bytes) and ss = 16*sp_stride (0 -> row_bytes). Every address,
// stride and length is byte-granular: the realigner shifts bytes between the DDR beat grid and
// the scratchpad line grid, AXI bursts are beat aligned (ARSIZE/AWSIZE = 16 B) and partial
// beats use WSTRB (stores) or the scratchpad byte strobe (loads). DDR addresses wrap at 2^AW
// and scratchpad addresses at 2^21 bytes.
//
// AXI: INCR bursts of up to MAXB beats (<= 256), never crossing a 4 KiB boundary; up to MAXRD
// read bursts and MAXWR write bursts outstanding; one ID (in-order). Read data lands in an LDF-
// beat FIFO whose space is reserved when the burst is issued (RREADY is always 1). `quiesce`
// stops new AR/AW (for a soft reset drain); `axi_idle` = nothing outstanding on AXI.
// Errors: RRESP/BRESP != OKAY sets the sticky `err` (the command still runs to completion).
//
// Scratchpad port (SG7 adapts it to the physical buffers): line address = byte address >> 4.
//   write: sp_wvalid/sp_wready, sp_wbuf, sp_waddr, sp_wdata, sp_wstrb (byte enables)
//   read : sp_arvalid/sp_arready, sp_arbuf, sp_araddr  ->  sp_rvalid, sp_rdata, in request
//          order, any latency, no backpressure (the DMA only requests lines it has room for)
module dma_engine #(
    parameter int AW    = 32,
    parameter int IDW   = 4,
    parameter int MAXB  = 256,   // max beats per burst (power of two, 1..256)
    parameter int LDF   = 512,   // load data FIFO depth (power of two, >= MAXB)
    parameter int STF   = 16,    // store line FIFO depth (power of two)
    parameter int MAXRD = 4,
    parameter int MAXWR = 4,
    parameter int NCMD  = 4      // commands in the engine (power of two >= 2); 1 command = no pipelining
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            quiesce,
    // command port
    input  logic            cmd_valid,
    output logic            cmd_ready,
    input  logic [127:0]    cmd_word,
    output logic            done,
    output logic            err,
    output logic            busy,
    output logic            axi_idle,
    output logic            bytes_valid,   // perf: a row of `bytes` bytes completed
    output logic [16:0]     bytes,
    // AXI4 master
    output logic [IDW-1:0]  m_axi_awid,
    output logic [AW-1:0]   m_axi_awaddr,
    output logic [7:0]      m_axi_awlen,
    output logic [2:0]      m_axi_awsize,
    output logic [1:0]      m_axi_awburst,
    output logic            m_axi_awlock,
    output logic [3:0]      m_axi_awcache,
    output logic [2:0]      m_axi_awprot,
    output logic            m_axi_awvalid,
    input  logic            m_axi_awready,
    output logic [127:0]    m_axi_wdata,
    output logic [15:0]     m_axi_wstrb,
    output logic            m_axi_wlast,
    output logic            m_axi_wvalid,
    input  logic            m_axi_wready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [IDW-1:0]  m_axi_bid,
    input  logic [1:0]      m_axi_bresp,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic            m_axi_bvalid,
    output logic            m_axi_bready,
    output logic [IDW-1:0]  m_axi_arid,
    output logic [AW-1:0]   m_axi_araddr,
    output logic [7:0]      m_axi_arlen,
    output logic [2:0]      m_axi_arsize,
    output logic [1:0]      m_axi_arburst,
    output logic            m_axi_arlock,
    output logic [3:0]      m_axi_arcache,
    output logic [2:0]      m_axi_arprot,
    output logic            m_axi_arvalid,
    input  logic            m_axi_arready,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [IDW-1:0]  m_axi_rid,
    input  logic [1:0]      m_axi_rresp,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [127:0]    m_axi_rdata,
    input  logic            m_axi_rlast,
    input  logic            m_axi_rvalid,
    output logic            m_axi_rready,
    // scratchpad line port
    output logic            sp_wvalid,
    input  logic            sp_wready,
    output logic [2:0]      sp_wbuf,
    output logic [16:0]     sp_waddr,
    output logic [127:0]    sp_wdata,
    output logic [15:0]     sp_wstrb,
    output logic            sp_arvalid,
    input  logic            sp_arready,
    output logic [2:0]      sp_arbuf,
    output logic [16:0]     sp_araddr,
    input  logic            sp_rvalid,
    input  logic [127:0]    sp_rdata
);
    // ---------------- command decode (generated field map) ----------------
    /* verilator lint_off UNUSEDSIGNAL */
    logic        f_is_store;
    logic [2:0]  f_buf_l, f_buf_s;
    logic [20:0] f_sp_l, f_sp_s;
    logic [31:0] f_ddr_l, f_ddr_s;
    logic [12:0] f_rows_l, f_rows_s;
    logic [16:0] f_rb_l, f_rb_s;
    logic [23:0] f_ds_l, f_ds_s;
    logic [11:0] f_ss_l, f_ss_s;
    /* verilator lint_on UNUSEDSIGNAL */
    /* verilator lint_off PINMISSING */
    cmd_fields u_f (
        .w(cmd_word), .is_dma_store(f_is_store),
        .dma_load_sp_buf(f_buf_l), .dma_load_sp_addr(f_sp_l), .dma_load_ddr_addr(f_ddr_l),
        .dma_load_rows(f_rows_l), .dma_load_row_bytes(f_rb_l), .dma_load_ddr_stride(f_ds_l),
        .dma_load_sp_stride(f_ss_l),
        .dma_store_sp_buf(f_buf_s), .dma_store_sp_addr(f_sp_s), .dma_store_ddr_addr(f_ddr_s),
        .dma_store_rows(f_rows_s), .dma_store_row_bytes(f_rb_s), .dma_store_ddr_stride(f_ds_s),
        .dma_store_sp_stride(f_ss_s));
    /* verilator lint_on PINMISSING */

    // ---------------- descriptors ----------------
    localparam int DW  = 1 + 3 + 17 + 25 + 17 + 13 + AW + 21;
    localparam int QD  = (NCMD < 2) ? 2 : NCMD;
    localparam int QCB = $clog2(QD + 1);
    logic          d_st;
    logic [2:0]    d_buf;
    logic [16:0]   d_rb, d_ss;
    logic [24:0]   d_ds;
    logic [12:0]   d_rows;
    logic [AW-1:0] d_ddr;
    logic [20:0]   d_sp;
    always_comb begin
        d_st   = f_is_store;
        d_buf  = f_is_store ? f_buf_s : f_buf_l;
        d_rb   = f_is_store ? f_rb_s : f_rb_l;
        d_ds   = ((f_is_store ? f_ds_s : f_ds_l) == '0) ? 25'(d_rb) : {1'b0, (f_is_store ? f_ds_s : f_ds_l)};
        d_ss   = ((f_is_store ? f_ss_s : f_ss_l) == '0) ? d_rb : {1'b0, f_is_store ? f_ss_s : f_ss_l, 4'b0};
        d_rows = f_is_store ? f_rows_s : f_rows_l;
        d_ddr  = AW'(f_is_store ? f_ddr_s : f_ddr_l);
        d_sp   = f_is_store ? f_sp_s : f_sp_l;
    end

    logic           aq_valid, aq_pop, rq_valid, rq_pop, aq_in_ready, rq_in_ready, cmd_take;
    logic [DW-1:0]  aq_data, rq_data;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [QCB-1:0] aq_count;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [QCB-1:0] rq_count;

    // address-stage descriptor (head of aq)
    logic          q_st;
    logic [16:0]   q_rb;
    logic [24:0]   q_ds;
    logic [12:0]   q_rows;
    logic [AW-1:0] q_ddr;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [2:0]    q_buf_u;
    logic [16:0]   q_ss_u;
    logic [20:0]   q_sp_u;
    /* verilator lint_on UNUSEDSIGNAL */
    assign {q_st, q_buf_u, q_rb, q_ds, q_ss_u, q_rows, q_ddr, q_sp_u} = aq_data;
    // data-stage descriptor (head of rq)
    logic          st;
    logic [2:0]    c_buf;
    logic [16:0]   c_rb;
    logic [24:0]   c_ds;
    logic [16:0]   c_ss;
    logic [12:0]   c_rows;
    logic [AW-1:0] c_ddr;
    logic [20:0]   c_sp;
    assign {st, c_buf, c_rb, c_ds, c_ss, c_rows, c_ddr, c_sp} = rq_data;

    // a STORE only into an empty engine, nothing behind a STORE, at most NCMD commands
    assign cmd_ready = rq_in_ready && aq_in_ready && 32'(rq_count) < NCMD &&
                       (rq_count == '0 || (!f_is_store && !st));
    assign cmd_take  = cmd_valid && cmd_ready;
    assign busy      = rq_valid;

    fifo_sync #(.W(DW), .DEPTH(QD)) u_aq (
        .clk(clk), .rst_n(rst_n), .in_valid(cmd_take), .in_ready(aq_in_ready),
        .in_data({d_st, d_buf, d_rb, d_ds, d_ss, d_rows, d_ddr, d_sp}),
        .out_valid(aq_valid), .out_ready(aq_pop), .out_data(aq_data), .count(aq_count));
    fifo_sync #(.W(DW), .DEPTH(QD)) u_rq (
        .clk(clk), .rst_n(rst_n), .in_valid(cmd_take), .in_ready(rq_in_ready),
        .in_data({d_st, d_buf, d_rb, d_ds, d_ss, d_rows, d_ddr, d_sp}),
        .out_valid(rq_valid), .out_ready(rq_pop), .out_data(rq_data), .count(rq_count));

    // ---------------- address loop (AR for loads, AW for stores) ----------------
    logic          a_act;       // a descriptor is loaded in the address stage
    logic          a_st;
    logic [16:0]   a_rb;
    logic [24:0]   a_ds;
    logic [12:0]   a_rows;      // rows left to issue
    logic [AW-1:0] a_row;       // DDR byte address of the current row
    logic          a_inrow;     // current row's beats are being issued
    logic [AW-1:0] a_addr;      // next beat address (16-byte aligned)
    logic [12:0]   a_left;      // beats left in the current row
    logic [8:0]    a_burst;
    logic          a_issue;
    logic [8:0]    a_to4k;
    localparam int RB = $clog2(MAXRD + 1);
    localparam int WB = $clog2(MAXWR + 1);
    localparam int LB = $clog2(LDF + 1);
    logic [RB-1:0] rd_out;      // read bursts outstanding
    logic [WB-1:0] wr_out;      // write bursts awaiting B
    logic [LB-1:0] ld_resv;     // load FIFO beats reserved (in flight + buffered)
    logic          ax_pending;  // AR/AW valid held

    assign a_to4k = 9'd256 - {1'b0, a_addr[11:4]};
    always_comb begin
        logic [12:0] m;
        m = a_left;
        if (13'(a_to4k) < m) m = 13'(a_to4k);
        if (13'(MAXB) < m)   m = 13'(MAXB);
        a_burst = 9'(m);
    end

    logic          wl_in_ready;
    logic a_room;
    assign a_room  = a_st ? (32'(wr_out) < MAXWR && wl_in_ready)
                          : (32'(rd_out) < MAXRD && 32'(ld_resv) + 32'(a_burst) <= LDF);
    assign a_issue = a_act && a_inrow && !ax_pending && !quiesce && a_room;
    assign aq_pop  = !a_act && aq_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            a_act      <= 1'b0;
            a_rows     <= '0;
            a_inrow    <= 1'b0;
            ax_pending <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_awvalid <= 1'b0;
        end else begin
            if (aq_pop) begin
                a_act   <= 1'b1;
                a_st    <= q_st;
                a_rb    <= q_rb;
                a_ds    <= q_ds;
                a_rows  <= q_rows;
                a_row   <= q_ddr;
                a_inrow <= 1'b0;
            end
            // start the next row; the stage frees once the command's last row is issued
            if (a_act && !a_inrow) begin
                if (a_rows != '0) begin
                    a_inrow <= 1'b1;
                    a_addr  <= {a_row[AW-1:4], 4'b0};
                    a_left  <= 13'((18'(a_row[3:0]) + 18'(a_rb) + 18'd15) >> 4);
                    a_rows  <= a_rows - 1'b1;
                    a_row   <= a_row + AW'(a_ds);
                end else begin
                    a_act <= 1'b0;
                end
            end
            if (a_issue) begin
                ax_pending <= 1'b1;
                if (a_st) begin
                    m_axi_awvalid <= 1'b1;
                    m_axi_awaddr  <= a_addr;
                    m_axi_awlen   <= 8'(a_burst - 9'd1);
                end else begin
                    m_axi_arvalid <= 1'b1;
                    m_axi_araddr  <= a_addr;
                    m_axi_arlen   <= 8'(a_burst - 9'd1);
                end
                a_addr <= a_addr + (AW'(a_burst) << 4);
                a_left <= a_left - 13'(a_burst);
                if (a_left == 13'(a_burst)) a_inrow <= 1'b0;
            end
            if (m_axi_arvalid && m_axi_arready) begin m_axi_arvalid <= 1'b0; ax_pending <= 1'b0; end
            if (m_axi_awvalid && m_axi_awready) begin m_axi_awvalid <= 1'b0; ax_pending <= 1'b0; end
        end
    end

    assign m_axi_arid = '0;  assign m_axi_arsize = 3'd4; assign m_axi_arburst = 2'b01;
    assign m_axi_arlock = 1'b0; assign m_axi_arcache = 4'b0011; assign m_axi_arprot = 3'b000;
    assign m_axi_awid = '0;  assign m_axi_awsize = 3'd4; assign m_axi_awburst = 2'b01;
    assign m_axi_awlock = 1'b0; assign m_axi_awcache = 4'b0011; assign m_axi_awprot = 3'b000;

    // ---------------- realign / row-consumer loop ----------------
    logic          r_ld;        // the data stage holds the head of rq
    logic [12:0]   c_left;      // rows left to move through the realigner
    logic [AW-1:0] r_ddr;       // DDR byte address of the realigner's row
    logic [20:0]   r_sp;        // scratchpad byte address of the realigner's row
    logic          r_start, r_busy, r_done_row;
    logic          ra_in_valid, ra_in_ready, ra_out_valid, ra_out_ready;
    /* verilator lint_off UNUSEDSIGNAL */
    logic          ra_last;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [127:0]  ra_in_data, ra_out_data;
    logic [15:0]   ra_strb;
    logic [12:0]   ra_idx;
    logic [3:0]    ra_in_off, ra_out_off;
    logic [16:0]   r_line_base;  // scratchpad line of the realigner's row

    // ---------------- store: scratchpad read loop ----------------
    logic [12:0]   s_rows;
    logic [20:0]   s_row;
    logic          s_inrow;
    logic [16:0]   s_line;
    logic [12:0]   s_left;
    localparam int SB = $clog2(STF + 1);
    logic [SB-1:0] st_resv;

    // FIFOs
    logic          ld_out_valid, ld_out_ready;
    logic [127:0]  ld_out_data;
    logic          sf_out_valid, sf_out_ready;
    logic [127:0]  sf_out_data;
    /* verilator lint_off UNUSEDSIGNAL */
    logic          ld_in_ready, sf_in_ready;
    logic [$clog2(LDF+1)-1:0] ld_count;
    logic [$clog2(STF+1)-1:0] sf_count;
    logic [3:0]    wl_count;
    /* verilator lint_on UNUSEDSIGNAL */
    logic          wl_out_valid, wl_out_ready;
    logic [7:0]    wl_out_len;
    logic [7:0]    w_beat;

    // ---------------- outstanding counters, errors ----------------
    logic rd_last_hs, b_hs, ld_pop;
    assign rd_last_hs   = m_axi_rvalid && m_axi_rready && m_axi_rlast;
    assign b_hs         = m_axi_bvalid && m_axi_bready;
    assign m_axi_rready = 1'b1;     // space reserved at AR issue
    assign m_axi_bready = 1'b1;
    assign ld_pop       = ld_out_valid && ld_out_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rd_out  <= '0;
            wr_out  <= '0;
            ld_resv <= '0;
            err     <= 1'b0;
        end else begin
            rd_out  <= rd_out + RB'(m_axi_arvalid && m_axi_arready) - RB'(rd_last_hs);
            wr_out  <= wr_out + WB'(m_axi_awvalid && m_axi_awready) - WB'(b_hs);
            ld_resv <= ld_resv + ((a_issue && !a_st) ? LB'(a_burst) : '0) - LB'(ld_pop);
            if (m_axi_rvalid && m_axi_rresp[1]) err <= 1'b1;
            if (b_hs && m_axi_bresp[1])         err <= 1'b1;
        end
    end
    // also no scratchpad read in flight (a soft reset must not catch a late sp_rvalid)
    logic [SB-1:0] sp_rd_out;
    always_ff @(posedge clk) begin
        if (!rst_n) sp_rd_out <= '0;
        else        sp_rd_out <= sp_rd_out + SB'(sp_arvalid && sp_arready) - SB'(sp_rvalid);
    end
    assign axi_idle = (rd_out == '0) && (wr_out == '0) && !ax_pending && !wl_out_valid && (sp_rd_out == '0);

    fifo_sync #(.W(128), .DEPTH(LDF)) u_ldf (
        .clk(clk), .rst_n(rst_n), .in_valid(m_axi_rvalid), .in_ready(ld_in_ready),
        .in_data(m_axi_rdata), .out_valid(ld_out_valid), .out_ready(ld_out_ready),
        .out_data(ld_out_data), .count(ld_count));

    // the data stage takes the head descriptor one cycle after it reaches the head of rq
    logic r_load;
    assign r_load = rq_valid && !r_ld;

    // ---------------- store: scratchpad line reads ----------------
    logic sp_req_hs;
    assign sp_req_hs  = sp_arvalid && sp_arready;
    assign sp_arvalid = r_ld && st && s_inrow && (32'(st_resv) < STF);
    assign sp_arbuf   = c_buf;
    assign sp_araddr  = s_line;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_rows  <= '0;
            s_inrow <= 1'b0;
            st_resv <= '0;
        end else begin
            if (r_load) begin
                s_rows  <= st ? c_rows : '0;
                s_row   <= c_sp;
                s_inrow <= 1'b0;
            end else if (r_ld && st && !s_inrow && s_rows != '0) begin
                s_inrow <= 1'b1;
                s_line  <= s_row[20:4];
                s_left  <= 13'((18'(s_row[3:0]) + 18'(c_rb) + 18'd15) >> 4);
                s_rows  <= s_rows - 1'b1;
                s_row   <= s_row + 21'(c_ss);
            end
            if (sp_req_hs) begin
                s_line <= s_line + 1'b1;
                s_left <= s_left - 1'b1;
                if (s_left == 13'd1) s_inrow <= 1'b0;
            end
            st_resv <= st_resv + SB'(sp_req_hs) - SB'(sf_out_valid && sf_out_ready);
        end
    end

    fifo_sync #(.W(128), .DEPTH(STF)) u_stf (
        .clk(clk), .rst_n(rst_n), .in_valid(sp_rvalid), .in_ready(sf_in_ready),
        .in_data(sp_rdata), .out_valid(sf_out_valid), .out_ready(sf_out_ready),
        .out_data(sf_out_data), .count(sf_count));

    // ---------------- realigner row loop ----------------
    assign ra_in_off  = st ? r_sp[3:0] : r_ddr[3:0];
    assign ra_out_off = st ? r_ddr[3:0] : r_sp[3:0];
    assign r_start    = r_ld && !r_busy && c_left != '0;

    dma_realign u_ra (
        .clk(clk), .rst_n(rst_n), .start(r_start), .in_off(ra_in_off), .out_off(ra_out_off),
        .len(c_rb), .busy(r_busy),
        .in_valid(ra_in_valid), .in_ready(ra_in_ready), .in_data(ra_in_data),
        .out_valid(ra_out_valid), .out_ready(ra_out_ready), .out_data(ra_out_data),
        .out_strb(ra_strb), .out_idx(ra_idx), .out_last(ra_last), .row_done(r_done_row));

    assign ra_in_valid  = st ? sf_out_valid : ld_out_valid;
    assign ra_in_data   = st ? sf_out_data  : ld_out_data;
    assign sf_out_ready = r_ld && st && ra_in_ready;
    assign ld_out_ready = r_ld && !st && ra_in_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            r_ld   <= 1'b0;
            c_left <= '0;
        end else if (r_load) begin
            r_ld   <= 1'b1;
            c_left <= c_rows;
            r_ddr  <= c_ddr;
            r_sp   <= c_sp;
        end else if (rq_pop) begin
            r_ld   <= 1'b0;
        end else if (r_start) begin
            r_line_base <= r_sp[20:4];
        end else if (r_done_row) begin
            c_left <= c_left - 1'b1;
            r_ddr  <= r_ddr + AW'(c_ds);
            r_sp   <= r_sp + 21'(c_ss);
        end
    end
    assign bytes_valid = r_done_row;
    assign bytes       = c_rb;

    // load: realigner -> scratchpad writes
    assign sp_wvalid = r_ld && !st && ra_out_valid;
    assign sp_wbuf   = c_buf;
    assign sp_waddr  = r_line_base + 17'(ra_idx);
    assign sp_wdata  = ra_out_data;
    assign sp_wstrb  = ra_strb;

    // store: AW burst lengths -> W beats with WLAST
    fifo_sync #(.W(8), .DEPTH(8)) u_wl (
        .clk(clk), .rst_n(rst_n), .in_valid(m_axi_awvalid && m_axi_awready), .in_ready(wl_in_ready),
        .in_data(m_axi_awlen), .out_valid(wl_out_valid), .out_ready(wl_out_ready),
        .out_data(wl_out_len), .count(wl_count));
    assign m_axi_wvalid = r_ld && st && ra_out_valid && wl_out_valid;
    assign m_axi_wdata  = ra_out_data;
    assign m_axi_wstrb  = ra_strb;
    assign m_axi_wlast  = (w_beat == wl_out_len);
    assign wl_out_ready = m_axi_wvalid && m_axi_wready && m_axi_wlast;
    always_ff @(posedge clk) begin
        if (!rst_n) w_beat <= '0;
        else if (m_axi_wvalid && m_axi_wready) w_beat <= m_axi_wlast ? 8'd0 : w_beat + 1'b1;
    end
    assign ra_out_ready = st ? (wl_out_valid && m_axi_wready) : sp_wready;

    // ---------------- completion ----------------
    // LOAD: every row through the realigner (all its data arrived, so its bursts are complete).
    // STORE (alone in the engine): also every AW issued and every write response back.
    assign done   = r_ld && c_left == '0 && !r_busy &&
                    (!st || (!a_act && !aq_valid && !ax_pending && wr_out == '0 && !wl_out_valid));
    assign rq_pop = done;

`ifndef SYNTHESIS
    always_ff @(posedge clk) begin
        if (rst_n && m_axi_rvalid && !ld_in_ready) $error("dma_engine: load FIFO overflow");
        if (rst_n && sp_rvalid && !sf_in_ready)    $error("dma_engine: store FIFO overflow");
    end
`endif
endmodule
