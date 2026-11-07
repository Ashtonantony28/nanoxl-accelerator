// dma_realign.sv -- byte realigner for one DMA row (16-byte beats in, 16-byte beats out).
//
// A row is `len` bytes (1..65536) that start at byte `in_off` of the first input beat and must
// start at byte `out_off` of the first output beat. The unit consumes
//   n_in  = ceil((in_off  + len) / 16) input beats and produces
//   n_out = ceil((out_off + len) / 16) output beats, each with a byte strobe that is 1 exactly
// on the row's bytes. With d = in_off - out_off, output beat j holds input-stream bytes
// [16j + d, 16j + d + 15]; for d < 0 a virtual zero beat is prepended, so in all cases
// out[j] = ({X[j+1], X[j]} >> 8e)[127:0] with e = d mod 16 and X the (extended) input stream.
// After the first beat of a row the unit moves one beat per cycle. `start` is accepted when
// idle (= !busy); `row_done` pulses with the last output beat of the row.
module dma_realign (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          start,
    input  logic [3:0]    in_off,
    input  logic [3:0]    out_off,
    input  logic [16:0]   len,       // 1..65536 (0 is never presented)
    output logic          busy,
    input  logic          in_valid,
    output logic          in_ready,
    input  logic [127:0]  in_data,
    output logic          out_valid,
    input  logic          out_ready,
    output logic [127:0]  out_data,
    output logic [15:0]   out_strb,
    output logic [12:0]   out_idx,   // index j of the output beat within the row
    output logic          out_last,  // last beat of the row
    output logic          row_done
);
    logic [12:0]  nx, n_out, cin, cout;
    logic [3:0]   e, end_off, oo;
    logic         have_prev;
    logic [127:0] prev;
    logic [255:0] cat;
    logic         need_in;
    // the first X beat is virtual when in_off < out_off (d < 0), detected by nx > n_in
    logic        virt_first, load_first;
    logic [12:0] n_in_q;

    // row geometry, latched on start
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
        end else if (start && !busy) begin
            logic [17:0] in_end, out_end;
            logic signed [4:0] d;
            in_end  = 18'(in_off)  + 18'(len) + 18'd15;
            out_end = 18'(out_off) + 18'(len) + 18'd15;
            d       = $signed({1'b0, in_off}) - $signed({1'b0, out_off});
            busy    <= 1'b1;
            n_out   <= 13'(out_end >> 4);
            nx      <= 13'(in_end >> 4) + 13'(d < 0);
            e       <= 4'(d);
            oo      <= out_off;
            end_off <= 4'(out_off + 4'(len - 17'd1));   // byte of the row's last byte in its beat
        end else if (row_done) begin
            busy <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n || (start && !busy)) begin
            have_prev <= 1'b0;
            cin       <= '0;
            cout      <= '0;
        end else if (busy) begin
            if (load_first) begin
                have_prev <= 1'b1;
                prev      <= virt_first ? '0 : in_data;
                cin       <= 13'd1;
            end else if (out_valid && out_ready) begin
                cout <= cout + 1'b1;
                if (need_in) begin
                    prev <= in_data;
                    cin  <= cin + 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (start && !busy) n_in_q <= 13'((18'(in_off) + 18'(len) + 18'd15) >> 4);
    end
    assign virt_first = (nx != n_in_q);
    assign load_first = busy && !have_prev && (virt_first || in_valid);

    assign need_in   = (cin < nx);
    assign cat       = {need_in ? in_data : 128'b0, prev};
    assign out_data  = 128'(cat >> {e, 3'b000});
    assign out_valid = busy && have_prev && (!need_in || in_valid);
    assign in_ready  = busy && ((!have_prev && !virt_first) || (have_prev && need_in && out_ready));
    assign out_idx   = cout;
    assign out_last  = (cout == n_out - 13'd1);
    assign row_done  = out_valid && out_ready && out_last;

    always_comb begin
        for (int b = 0; b < 16; b++) begin
            logic lo_ok, hi_ok;
            lo_ok = (cout != 13'd0) || (4'(b) >= oo);
            hi_ok = !out_last || (4'(b) <= end_off);
            out_strb[b] = lo_ok && hi_ok;
        end
    end
endmodule
