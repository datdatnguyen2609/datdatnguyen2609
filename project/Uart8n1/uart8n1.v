module uart8n1 #(
    parameter CLOCK_RATE   = 100000000, // board clock (default 100MHz)
    parameter BAUD_RATE    = 9600,
    parameter TURBO_FRAMES = 0          // see Uart8Transmitter
)(
    input wire clk, // board clock (*note: at the {CLOCK_RATE} rate)

    // rx interface
    input wire rxEn,
    input wire rx,
    output wire rxBusy,
    output wire rxDone,
    output wire rxErr,
    output wire [7:0] out,

    // tx interface
    input wire txEn,
    input wire txStart,
    input wire [7:0] in,
    output wire txBusy,
    output wire txDone,
    output wire tx
);

// this value cannot be changed in the current implementation
parameter RX_OVERSAMPLE_RATE = 16;

wire rxClk;
wire txClk;

BaudRateGenerator #(
    .CLOCK_RATE(CLOCK_RATE),
    .BAUD_RATE(BAUD_RATE),
    .RX_OVERSAMPLE_RATE(RX_OVERSAMPLE_RATE)
) generatorInst (
    .clk(clk),
    .rxClk(rxClk),
    .txClk(txClk)
);

Uart8Receiver rxInst (
    .clk(rxClk),
    .en(rxEn),
    .in(rx),
    .busy(rxBusy),
    .done(rxDone),
    .err(rxErr),
    .out(out)
);

Uart8Transmitter #(
    .TURBO_FRAMES(TURBO_FRAMES)
) txInst (
    .clk(txClk),
    .en(txEn),
    .start(txStart),
    .in(in),
    .busy(txBusy),
    .done(txDone),
    .out(tx)
);

endmodule

// states of state machine
`define RESET     3'b000
`define IDLE      3'b001
`define START_BIT 3'b010
`define DATA_BITS 3'b011
`define STOP_BIT  3'b100
`define READY     3'b101 // receiver only

module BaudRateGenerator #(
    parameter CLOCK_RATE         = 100000000, // board clock (default 100MHz)
    parameter BAUD_RATE          = 9600,
    parameter RX_OVERSAMPLE_RATE = 16
)(
    input wire clk,   // board clock (*note: at the {CLOCK_RATE} rate)
    output reg rxClk, // baud rate for rx
    output reg txClk  // baud rate for tx
);

localparam RX_ACC_MAX   = CLOCK_RATE / (2 * BAUD_RATE * RX_OVERSAMPLE_RATE);
localparam TX_ACC_MAX   = CLOCK_RATE / (2 * BAUD_RATE);
localparam RX_ACC_WIDTH = $clog2(RX_ACC_MAX);
localparam TX_ACC_WIDTH = $clog2(TX_ACC_MAX);

reg [RX_ACC_WIDTH-1:0] rx_counter = 0;
reg [TX_ACC_WIDTH-1:0] tx_counter = 0;

initial begin
    rxClk = 1'b0;
    txClk = 1'b0;
end

always @(posedge clk) begin
    // rx clock
    if (rx_counter == RX_ACC_MAX[RX_ACC_WIDTH-1:0]) begin
        rx_counter <= 0;
        rxClk      <= ~rxClk;
    end else begin
        rx_counter <= rx_counter + 1'b1;
    end

    // tx clock
    if (tx_counter == TX_ACC_MAX[TX_ACC_WIDTH-1:0]) begin
        tx_counter <= 0;
        txClk      <= ~txClk;
    end else begin
        tx_counter <= tx_counter + 1'b1;
    end
end

endmodule

module Uart8Transmitter #(
    parameter TURBO_FRAMES = 0
)(
    input wire clk,      // baud rate
    input wire en,
    input wire start,    // start transmission
    input wire [7:0] in, // parallel data to transmit
    output reg busy,     // transmit is in progress
    output reg done,     // end of transmission
    output reg out       // tx line for serial data
);

reg [2:0] state     = `RESET;
reg [7:0] in_data   = 8'b0; // shift reg for the data to transmit serially
reg [2:0] bit_index = 3'b0; // index for 8-bit data

/*
 * Disable at any time in the flow
 */
always @(posedge clk) begin
    if (!en) begin
        state <= `RESET;
    end
end

/*
 * State machine
 */
always @(posedge clk) begin
    case (state)
        `RESET: begin
            // state variables
            bit_index <= 3'b0;
            // outputs
            busy      <= 1'b0;
            done      <= 1'b0;
            out       <= 1'b1; // drive the line high for IDLE state
            // next state
            if (en) begin
                state <= `IDLE;
            end
        end

        `IDLE: begin
            if (start) begin
                in_data <= in; // register the input data
                state   <= `START_BIT;
            end
        end

        `START_BIT: begin
            bit_index <= 3'b0;
            busy      <= 1'b1;
            done      <= 1'b0;
            out       <= 1'b0; // send the space output, aka start bit (low)
            state     <= `DATA_BITS;
        end

        `DATA_BITS: begin // take 8 clock cycles for data bits to be sent
            // grab each input bit using a shift register: the hardware
            // realization is simple compared to routing the access
            // dynamically, i.e. using in_data[bit_index]
            in_data   <= { 1'b0, in_data[7:1] };
            out       <= in_data[0];
            // manage the state transition
            bit_index <= bit_index + 3'b1;
            if (&bit_index) begin
                // bit_index wraps around to zero
                state <= `STOP_BIT;
            end
        end

        `STOP_BIT: begin
            done              <= 1'b1; // signal the transmission stop
            out               <= 1'b1; // transition to mark state output (high)
            if (start) begin
                if (done == 1'b0) begin // this distinguishes 2 sub-states
                    in_data   <= in; // register new input data
                    if (TURBO_FRAMES) begin
                        state <= `START_BIT; // go direct to transmit
                    end else begin
                        state <= `STOP_BIT; // keep mark state one extra cycle
                    end
                end else begin // there was extra cycle within this state
                    done      <= 1'b0;
                    state     <= `START_BIT; // now go to transmit
                end
            end else begin
                state         <= `RESET;
            end
        end

        default: begin
            state <= `RESET;
        end
    endcase
end

endmodule

module Uart8Receiver (
    input wire clk,      // rx data sampling rate
    input wire en,
    input wire in,       // rx line
    output reg busy,     // transaction is in progress
    output reg done,     // end of transaction
    output reg err,      // error while receiving data
    output reg [7:0] out // the received data assembled in parallel form
);


reg [2:0] state          = `RESET;
reg [1:0] in_reg         = 2'b0; // shift reg for input signal conditioning
reg [4:0] in_hold_reg    = 5'b0; // shift reg for signal hold time checks
reg [3:0] sample_count   = 4'b0; // count ticks for 16x oversample
reg [4:0] out_hold_count = 5'b0; // count ticks before clearing output data
reg [2:0] bit_index      = 3'b0; // index for 8-bit data
reg [7:0] received_data  = 8'b0; // shift reg for the deserialized data
wire in_sample;
wire [3:0] in_prior_hold_reg;
wire [3:0] in_current_hold_reg;

/*
 * Double-register the incoming data:
 *
 * This prevents metastability problems crossing into rx clock domain
 *
 * After registering, only the in_sample wire is to be accessed - the
 *   earlier, unconditioned signal {in} must be ignored
 */
always @(posedge clk) begin
    in_reg <= { in_reg[0], in };
end

assign in_sample = in_reg[1];

/*
 * Track the incoming data for 4 rx {clk} ticks + 1, to be able to enforce a
 *   minimum hold time of 4 {clk} ticks for any rx signal
 */
always @(posedge clk) begin
    in_hold_reg <= { in_hold_reg[3:1], in_sample, in_reg[0] };
end

assign in_prior_hold_reg   = in_hold_reg[4:1];
assign in_current_hold_reg = in_hold_reg[3:0];

/*
 * End the validity of output data after precise time of one serial bit cycle:
 *
 * Output signals from this module might as well be consistent with input
 *   rate, which is the baud rate
 *
 * This hold is for the case when detection of a next transmit cut the
 *   prior stop and ready transitions short; i.e. IDLE state has been entered
 *   direct from STOP_BIT state or READY state
 */
always @(posedge clk) begin
    if (|out_hold_count) begin
        out_hold_count     <= out_hold_count + 5'b1;
        if (out_hold_count == 5'b10000) begin // reached 16 -
            // timed output interval ends
            out_hold_count <= 5'b0;
            done           <= 1'b0;
            out            <= 8'b0;
        end
    end
end

/*
 * Disable at any time in the flow
 */
always @(posedge clk) begin
    if (!en) begin
        state <= `RESET;
    end
end

/*
 * State machine
 */
always @(posedge clk) begin
    case (state)
        `RESET: begin
            // state variables
            sample_count   <= 4'b0;
            out_hold_count <= 5'b0;
            received_data  <= 8'b0;
            // outputs
            busy           <= 1'b0;
            done           <= 1'b0;
            if (en && err && !in_sample) begin // in error condition already -
                // leave the output uninterrupted
                err        <= 1'b1;
            end else begin
                err        <= 1'b0;
            end
            out            <= 8'b0; // output parallel data only during {done}
            // next state
            if (en) begin
                state      <= `IDLE;
            end
        end

        `IDLE: begin
            /*
             * Accept low-going input as the trigger to start:
             *
             * Count from the first low sample, and sample again at the
             *   mid-point of a full baud interval to accept the low signal
             *
             * Then start the count for the proceeding full baud intervals
             */
            if (!in_sample) begin
                if (sample_count == 4'b0) begin
                    if (&in_prior_hold_reg || done && !err) begin
                        // meets the preceding min high hold time -
                        // note that {done} && !{err} encodes the fact that
                        // the min hold time was met earlier in STOP_BIT state
                        // or READY state
                        sample_count  <= 4'b1;
                        err           <= 1'b0;
                    end else begin
                        // this was a false start -
                        // remain in IDLE state with sample_count zero
                        err           <= 1'b1;
                    end
                end else begin
                    sample_count      <= sample_count + 4'b1;
                    if (sample_count == 4'b1100) begin // reached 12
                        // start signal meets an additional hold time
                        // of >= 4 rx ticks after its own mid-point -
                        // start new full interval count but from the mid-point
                        sample_count  <= 4'b0100;
                        busy          <= 1'b1;
                        err           <= 1'b0;
                        state         <= `START_BIT;
                    end
                end
            end else if (|sample_count) begin
                // bit did not remain low while waiting till 8 then 12 -
                // remain in IDLE state
                sample_count          <= 4'b0;
                received_data         <= 8'b0;
                err                   <= 1'b1;
            end
        end

        `START_BIT: begin
            /*
             * Wait one full baud interval to the mid-point of first bit
             */
            sample_count      <= sample_count + 4'b1;
            if (&sample_count) begin // reached 15
                // sample_count wraps around to zero
                bit_index     <= 3'b1;
                received_data <= { in_sample, 7'b0 };
                out           <= 8'b0;
                state         <= `DATA_BITS;
            end
        end

        `DATA_BITS: begin
            /*
             * Take 8 baud intervals to receive serial data
             */
            sample_count      <= sample_count + 4'b1;
            if (&sample_count) begin // reached 15 - save one more bit of data
                // store the bit using a shift register: the hardware
                // realization is simple compared to routing the bit
                // dynamically, i.e. using received_data[bit_index]
                received_data <= { in_sample, received_data[7:1] };
                // manage the state transition
                bit_index     <= bit_index + 3'b1;
                if (&bit_index) begin
                    // bit_index wraps around to zero
                    // sample_count wraps around to zero
                    state     <= `STOP_BIT;
                end
            end
        end

        `STOP_BIT: begin
            /*
             * Accept the received data if input goes high:
             *
             * If stop signal condition(s) met, drive the {done} signal high
             *   for one bit cycle
             *
             * Otherwise drive the {err} signal high for one bit cycle
             *
             * Since this baud clock may not track the transmitter baud clock
             *   precisely in reality, accept the transition to handling the
             *   next start bit any time after the stop bit mid-point
             */
            sample_count               <= sample_count + 4'b1;
            if (sample_count[3]) begin // reached 8 to 15
                if (!in_sample) begin
                    // accept that transmit has completed only if the stop
                    // signal held for a time of >= 4 rx ticks before it
                    // changed to a start signal
                    if (sample_count == 4'b1000 &&
                            &in_prior_hold_reg) begin // meets the hold time
                        // can accept the transmitted data and output it
                        sample_count   <= 4'b0;
                        out_hold_count <= 5'b1;
                        done           <= 1'b1;
                        out            <= received_data;
                        state          <= `IDLE;
                    end else if (&sample_count) begin // reached 15
                        // bit did not go high or remain high -
                        // signal {err}, continuing until condition resolved
                        sample_count   <= 4'b0;
                        received_data  <= 8'b0;
                        busy           <= 1'b0;
                        err            <= 1'b1;
                        state          <= `IDLE;
                    end
                end else begin
                    if (&in_current_hold_reg) begin // meets min high hold time
                        // can accept the transmitted data and output it
                        sample_count   <= 4'b0;
                        done           <= 1'b1;
                        out            <= received_data;
                        state          <= `READY;
                    end else if (&sample_count) begin // reached 15
                        // did not meet min high hold time -
                        // signal {err} for this transmit
                        sample_count   <= 4'b0;
                        err            <= 1'b1;
                        state          <= `READY;
                    end
                end
            end
        end

        `READY: begin
            /*
             * Wait one full bit cycle to sustain the {out} data, the
             *   {done} signal or the {err} signal
             */
            sample_count              <= sample_count + 4'b1;
            if (!err && !in_sample || &sample_count) begin
                // check if this is the change to a start signal -
                // note that in this state, in_sample has met the min high
                // hold time any time it drops to low
                // (also in these cases, namely !{err} or tick 15 special case,
                //  signaling of {done} is in progress)
                if (&sample_count) begin // reached 15, last tick, and no error
                    // (signaling of {done} is now complete)
                    if (in_sample) begin
                        // not transitioning to start bit -
                        // sample_count wraps around to zero
                        received_data <= 8'b0;
                        busy          <= 1'b0;
                    end else begin
                        // transitioning to start bit -
                        // sustain the {busy} signal high
                        sample_count  <= 4'b1;
                    end
                    done              <= 1'b0;
                    out               <= 8'b0;
                    state             <= `IDLE;
                end else begin
                    // in_sample drops from high to low
                    // (signaling of {done} continues)
                    sample_count      <= 4'b1;
                    // continue the counting
                    out_hold_count    <= sample_count + 5'b00010;
                    state             <= `IDLE;
                end
            end else if (&sample_count[3:1]) begin // reached 14 -
                // additional tick 15 comes from transiting the READY state
                // to get to the RESET state
                if (err || !in_sample) begin
                    state             <= `RESET;
                end
                // otherwise, signaling of {done} is in progress (i.e. !{err}) -
                // in this case, on tick 15, will be checking if in_sample
                // dropped from high to low on the entry to IDLE state
            end
        end

        default: begin
            state <= `RESET;
        end
    endcase
end

endmodule
