//`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:         UDE-ES
// Designer:        AE
// Module Name:     Ethernet MAC-Controller mit IPv4 und UDP-Protokoll via IEEE 802.3
//                  MII-Interface mit Übertragungsrate: 100 Mbit/s
//                  
// Version:         1v1 
// Date:            10.03.2022
// Description:     TX mit CRC lauflähig
//                  RX fehlt noch
// Comments:        Bei DATA_LGTH kleiner als 18 Bytes wird das Ende vorm FCS mit Nullfolgen (PADDING) aufgefüllt
//                  MAC-Adresse auslesen im cmd mit getmac /v oder arp -a
//                  Übertragung: Preample (8 bytes) + Header (46 Bytes) + Daten (x Bytes) 
//                  Effizienz von 12,9% (x=8) bis 94,9% (x=1450)
//                  Angabe des SRC_MAC willkürlich, aber [47:32] = 'h24_4B  
//                  Angabe der SRC_IP mit 169.xxx.xxx.xxx --> 48'hA9_XX_XX_XX
//////////////////////////////////////////////////////////////////////////////////
`define IP_PARAMS_SRC_INTERNAL
//`define IP_PARAMS_DST_INTERNAL

/// --- Header of Ethernet frame.
/// *   [dst_mac] is the destination MAC address.
/// *   [src_mac] is the source MAC address.
/// *   [ether_type] indicates the protocol of the packet being sent.
typedef struct packed {
    logic [47:0] dst_mac;
    logic [47:0] src_mac;
    logic [15:0] ether_type;
} MacHeader;

/// --- IPv4 Header
///  0                   1                   2                   3
///  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
/// +---------------+---------------+---------------+---------------+
/// |Version|  IHL  |     Type      |          Total Length         |
/// +---------------+---------------+---------------+---------------+
/// |         Identification        |Flags|     Fragment Offset     |
/// +---------------+---------------+---------------+---------------+
/// | Time to Live  |    Protocol   |        Header Checksum        |
/// +---------------+---------------+---------------+---------------+
/// |                       Source IP Address                       |
/// +---------------+---------------+---------------+---------------+
/// |                    Destination IP Address                     |
/// +---------------+---------------+---------------+---------------+
/// *   [version] indicates the format of internet header.
/// *   [ihl] is the internet header length. This must be at least 5.
/// *   [type_of_service] is the type and quality of service desired.
/// *   [total_length] is the total length of packet including the header and
///     data.
/// *   [identification] is  assigned to help with assembling fragments.
/// *   [flags] is a set of control flags.
/// *   [fragment_offset] indicates where in a datagram the fragment belongs.
/// *   [time_to_live] the time (formally in seconds, practically in hops) that
///     the datagram is allowed to live.
/// *   [protocol] the next level protocol to use.
/// *   [header_checksum] is the checksum that verifies the validity of the
///     IP header.
/// *   [src_ip] is the source IPv4 address.
/// *   [dst_ip] is the destination IPv4 address.
typedef struct packed {
    logic [3:0] version;
    logic [3:0] ihl;
    logic [7:0] type_of_service;
    logic [15:0] total_length;
    logic [15:0] identification;
    logic [2:0] flags;
    logic [12:0] fragment_offset;
    logic [7:0] time_to_live;
    logic [7:0] protocol;
    logic [15:0] header_checksum;
    logic [31:0] src_ip;
    logic [31:0] dst_ip;
} IpHeader;

/// --- UDP Header
///  0      7 8     15 16    23 24    31
/// +--------+--------+--------+--------+
/// |   Source Port   |    Dest Port    |
/// +--------+--------+--------+--------+
/// |     Length      |    Checksum     |
/// +--------+--------+--------+--------+
/// *   [src_port] is the number of the source port.
/// *   [dst_port] is the number of the destination port.
/// *   [length] is the length of the packet in bytes inclufing this header.
/// *   [checksum] is the UDP checksum.
typedef struct packed {
    logic [15:0] src_port;
    logic [15:0] dst_port;
    logic [15:0] length;
    logic [15:0] checksum;
} UdpHeader;

module MII_ETH_PHY #(
    parameter byte_tx  = 12'd6,
    `ifdef IP_PARAMS_SRC_INTERNAL
        parameter SRC_PARAM_IP   = 32'hA9_00_00_01,
        parameter SRC_PARAM_MAC  = 48'h24_4B_FE_55_8B_BA,
        parameter SRC_PARAM_PORT = 16'h0400,
    `endif
    `ifdef IP_PARAMS_DST_INTERNAL
        parameter DST_PARAM_IP   = 32'hA9_FE_9D_DF,
        parameter DST_PARAM_MAC  = 48'h00_E0_4C_68_21_A0,
        parameter DST_PARAM_PORT = 16'h0400,
    `endif
    parameter int unsigned RST_CYC = 100_000,
    parameter int unsigned PWRUP_CYC = 20_000_000
    )(
    //---- Global control lines (internal)
    input wire CLK_100MHz,
    input wire nRST,
    `ifndef IP_PARAMS_SRC_INTERNAL
        input wire [47:0]   SRC_PARAM_MAC,
        input wire [31:0]   SRC_PARAM_IP,
        input wire [15:0]   SRC_PARAM_PORT,
    `endif
    `ifndef IP_PARAMS_DST_INTERNAL
        input wire [47:0]   DST_PARAM_MAC,
        input wire [31:0]   DST_PARAM_IP,
        input wire [15:0]   DST_PARAM_PORT,   
    `endif
    //--- Control lines for EthernetMAC
    output wire ETH_CLK_REF,
    output reg ETH_nRST,
    output reg ETH_RDY,
    //--- Control lines for Management Access
    output wire MD_EN,
    output wire MDIO,            
    output wire MDC,  
    //--- DataTX to EthernetMAC   
    input wire START_TX_FLAG,
    output reg TX_BUSY,
    input wire [8*byte_tx-'d1:0] DATA_TX, 
    input wire ETH_clkTX,
    output reg ETH_enTX,
    output reg [3:0] ETH_TXD    
);

    //========================== SETTINGS OF SEVERAL HEADERS =========================//
    localparam int unsigned SIZE_CNT = (RST_CYC > PWRUP_CYC) ? RST_CYC : PWRUP_CYC;
    // The number of nibbles in parts of the frame
    localparam int unsigned MIN_FRAME_NIBBLES    = 8'd128;
    localparam int unsigned PREAMBLE_SFD_NIBBLES = 8'd16;
    localparam int unsigned MAC_HEADER_NIBBLES   = 8'd28;
    localparam int unsigned IP_HEADER_NIBBLES    = 8'd40;
    localparam int unsigned UDP_HEADER_NIBBLES   = 8'd16;
    localparam int unsigned WORD_SIZE_NIBBLES    = 2 * byte_tx;
    localparam int unsigned PAD_SIZE_NIBBLES     = (8'd36 < WORD_SIZE_NIBBLES) ? 8'd0 : 8'd36 - WORD_SIZE_NIBBLES;
    localparam int unsigned FCS_NIBBLES          = 8'd8;
    localparam int unsigned GAP_NIBBLES          = 8'd24;

    localparam int unsigned ETHER_TYPE = 16'h0800;          // The Ethernet type for the Ethernet header. This value indicates that IPv4 is used.
    localparam int unsigned IP_VERSION = 4;                 // The IP version to use.
    localparam int unsigned IP_IHL = 5;                     // The IP header length.
    localparam int unsigned IP_TOS = 8'b010_1_1_0_0_0;      // The IP type of service (7-5: IP Precedence, 4: Delay, 3: Throughput, 2: Reliability, 1: Monetary Cost, 0: 0).
    localparam int unsigned IP_ID = 16'd0;                  // The IP fragment identification.
    localparam int unsigned IP_FLAGS = 3'b010;
    localparam int unsigned IP_FRAG_OFFSET = 16'd0;         // The IP fragmentation offset.
    localparam int unsigned IP_TTL = 8'h80;                 // The IP time to live.
    localparam int unsigned IP_PROTOCOL = 8'h11;            // The IP next level protocol to use. This is the User Datagram Protocol.  
    
    //========================== CRC calculation algorithm =========================//
    function logic [31:0] compute_crc(input logic [31:0] crc, input logic [3:0] data);
        localparam int unsigned POLYNOMIAL = 32'h04C11DB7;

        compute_crc = crc;
        for (int j = 0; j < 4; j++) begin
            compute_crc = {compute_crc[30:0], 1'b0} ^
                (data[j] == compute_crc[31] ? '0 : POLYNOMIAL);
        end
    endfunction

    function logic [31:0] swap_nibbles(input logic [31:0] data);
        swap_nibbles[28+:4] = data[24+:4];
        swap_nibbles[24+:4] = data[28+:4];
        swap_nibbles[20+:4] = data[16+:4];
        swap_nibbles[16+:4] = data[20+:4];
        swap_nibbles[12+:4] = data[8+:4];
        swap_nibbles[8+:4]  = data[12+:4];
        swap_nibbles[4+:4]  = data[0+:4];
        swap_nibbles[0+:4]  = data[4+:4];
    endfunction
        
    // From a [vector] grab a nibble at the given index. The order of the
    // nibbles in each byte that are selected is reversed.
    `define SLICE(vector, index) \
        vector[8* (index >> 1) + 4* ((~index) & 1)+:4]

    // This enum is used to track the progress of a state machine that writes
    // the data to the PHY.
    enum {
        INIT,
        PWRUP,
        MDX_INIT,
        TX_IDLE,
        PREPARE,
        IP_CHECKSUM,
        TX_SYNC,
        TX_MAC,
        TX_IP,
        TX_UDP,
        TX_DATA,
        TX_UDP_END,
        TX_PAD,
        TX_FCS,
        TX_GAP
    } state;   

    // The headers that are to be sent
    MacHeader mac_header;
    IpHeader ip_header;
    UdpHeader udp_header;
    
    logic [$clog2(SIZE_CNT)-'d1:0] cnt_flagTX;
    logic [31:0] fcs, checksum_temp;
    logic [5:0] padding_nibbles;
    logic [1:0] shift_clkTX, cnt_clk;
    
    wire clk_flag_tx;
    assign clk_flag_tx = !shift_clkTX[0] && shift_clkTX[1];    
    assign MD_EN    = (state == MDX_INIT);
    assign MDC      = (MD_EN) ? ETH_CLK_REF : 1'd0; 
    assign MDIO     = (MD_EN) ? 1'd1    : 1'bZ;
    assign ETH_CLK_REF = cnt_clk[1];
    
    //========================== STATE MACHINE FOR ETHERNET TX =========================//
    always_ff@(posedge CLK_100MHz) begin
        if (~nRST) begin
            ETH_nRST        <= 1'd0;
            ETH_TXD         <= 4'd0;
            ETH_enTX        <= 1'd0;
            mac_header      <= 1'd0;
            ip_header       <= 160'd0;
            udp_header      <= 64'd0;
            fcs             <= 32'd0;
            checksum_temp   <= 32'd0;
            cnt_flagTX      <= RST_CYC;
            TX_BUSY         <= 1'd0;
            ETH_RDY         <= 1'd0;
            padding_nibbles <= 'd0;
            shift_clkTX     <= 2'd0;
            state           <= INIT;
            cnt_clk         <= 2'd0;
        end else begin
            //CLK generation of 25 MHz
            cnt_clk     <= (state != INIT) ? cnt_clk + 2'd1 : 2'd0;
            //Schiebe-Register zur Synchronisierung der TX_CLK
            shift_clkTX <= {shift_clkTX[0], ETH_clkTX};
            // Run the state machine only when positive flag on synchronized ETH_clkTX                
            case (state)
            INIT: begin
                cnt_flagTX  <= (cnt_flagTX == 'd0) ? PWRUP_CYC    : cnt_flagTX - 'd1;
                ETH_nRST    <= (cnt_flagTX == 'd0) ? 1'd1         : 1'd0;
                state       <= (cnt_flagTX == 'd0) ? PWRUP        : state; 
                //ETH_nRST <= 1'd1;
                //state <= PWRUP;
            end
            // Wait for the PHY to properly power up
            PWRUP: begin
                cnt_flagTX  <= (cnt_flagTX == 'd0) ? 'd31       : cnt_flagTX - 'd1;
                ETH_RDY     <= (cnt_flagTX == 'd0) ? 1'd1       : 1'd0;
                state       <= (cnt_flagTX == 'd0) ? MDX_INIT   : state; 
            end
            // Send as soon as there is enough data in the FIFO
            MDX_INIT: begin
                cnt_flagTX  <= (cnt_flagTX == 'd0) ? 'd0        : cnt_flagTX - 'd1;
                state       <= (cnt_flagTX == 'd0) ? TX_IDLE    : state;
            end
            TX_IDLE: begin
                state    <= (START_TX_FLAG) ? PREPARE : state;
            end         
            PREPARE: begin
                // Construct the Ethernet header
                mac_header.dst_mac    <= DST_PARAM_MAC;
                mac_header.src_mac    <= SRC_PARAM_MAC;
                mac_header.ether_type <= ETHER_TYPE;
                // Construct the IP header
                ip_header.version         <= IP_VERSION;
                ip_header.ihl             <= IP_IHL;
                ip_header.type_of_service <= IP_TOS;
                ip_header.identification  <= IP_ID;
                ip_header.flags           <= IP_FLAGS;
                ip_header.fragment_offset <= IP_FRAG_OFFSET;
                ip_header.time_to_live    <= IP_TTL;
                ip_header.protocol        <= IP_PROTOCOL;
                ip_header.src_ip          <= SRC_PARAM_IP;
                ip_header.dst_ip          <= DST_PARAM_IP;
                ip_header.total_length    <= 16'd28 + byte_tx;
                ip_header.header_checksum <= 16'd0;
                checksum_temp             <= 32'd0;
                // Construct the UDP header
                udp_header.src_port <= SRC_PARAM_PORT;
                udp_header.dst_port <= DST_PARAM_PORT;
                udp_header.length   <= 16'd8 + byte_tx;
                udp_header.checksum <= 'd0; // Optional, left as 0
                // Others
                fcs <= 32'hFFFFFFFF;
                padding_nibbles <= PAD_SIZE_NIBBLES;
                cnt_flagTX  <= 'd1;
                state       <= IP_CHECKSUM; 
            end
            IP_CHECKSUM: begin
                 // Compute the IP header checksum
                if (cnt_flagTX == 'd1) begin
                    // Note that the header checksum field `ip_header[64+:16]` is not included.
                    checksum_temp <=
                        ip_header[144+:16] + // Version, IHL, ToS
                        ip_header[128+:16] + // Total length
                        ip_header[112+:16] + // Identification
                        ip_header[ 96+:16] + // Flags, Fragmentation offset
                        ip_header[ 80+:16] + // TTL, Protocol
                        ip_header[ 48+:16] + // Source IP Upper
                        ip_header[ 32+:16] + // Source IP Lower
                        ip_header[ 16+:16] + // Destination IP Upper
                        ip_header[  0+:16];  // Destination IP Lower
                    // Others
                    cnt_flagTX <= 'd0;
                    state <= state;
                end else begin
                    ip_header.header_checksum   <= ~(checksum_temp[31:16] + checksum_temp[15:0]);
                    cnt_flagTX <= PREAMBLE_SFD_NIBBLES - 'd1;
                    state <= TX_SYNC;
                end
            end
            // Send the preamble and SFD to the PHY
            TX_SYNC: begin
                TX_BUSY <= 1'd1;
                ETH_enTX <= 1'd1;            
                if(clk_flag_tx) begin
                    if (cnt_flagTX != 'd0) begin
                        state       <= state;
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        ETH_TXD     <= 4'h5;
                    end else begin
                        cnt_flagTX  <= MAC_HEADER_NIBBLES - 'd1;
                        state       <= TX_MAC;
                        ETH_TXD     <= 4'hD;
                    end
                end
            end
            TX_MAC: begin
                if(clk_flag_tx) begin
                    ETH_TXD <= `SLICE(mac_header, cnt_flagTX);
                    fcs     <= compute_crc(fcs, `SLICE(mac_header, cnt_flagTX));
                    if (cnt_flagTX != 'd0) begin
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        state       <= state;
                    end else begin
                        cnt_flagTX  <= IP_HEADER_NIBBLES - 'd1;
                        state       <= TX_IP;
                    end
                end
            end
            TX_IP: begin
                if(clk_flag_tx) begin
                    ETH_TXD <= `SLICE(ip_header, cnt_flagTX);
                    fcs     <= compute_crc(fcs, `SLICE(ip_header, cnt_flagTX));
                    if (cnt_flagTX != 'd0) begin
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        state       <= state;
                    end else begin
                        cnt_flagTX  <= UDP_HEADER_NIBBLES - 'd1;
                        state       <= TX_UDP;
                    end
                end             
            end
            TX_UDP: begin
                if(clk_flag_tx) begin
                    ETH_TXD <= `SLICE(udp_header, cnt_flagTX);
                    fcs     <= compute_crc(fcs, `SLICE(udp_header, cnt_flagTX));
                    if (cnt_flagTX != 'd0) begin
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        state       <= state;
                    end else begin
                        cnt_flagTX  <= (WORD_SIZE_NIBBLES == 'd0) ? PAD_SIZE_NIBBLES - 'd1 : WORD_SIZE_NIBBLES - 'd1; 
                        state       <= (WORD_SIZE_NIBBLES == 'd0) ? TX_PAD : TX_DATA;
                    end
                end
            end
            TX_DATA: begin
               if(clk_flag_tx) begin
                    ETH_TXD <= `SLICE(DATA_TX, cnt_flagTX);
                    if (cnt_flagTX != 'd0) begin
                        fcs         <= compute_crc(fcs, `SLICE(DATA_TX, cnt_flagTX));
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        state       <= state;
                    end else begin
                        fcs         <= (padding_nibbles == 'd0) ? swap_nibbles(compute_crc(fcs, `SLICE(DATA_TX, cnt_flagTX))) : compute_crc(fcs, `SLICE(DATA_TX, cnt_flagTX));
                        cnt_flagTX  <= (padding_nibbles == 'd0) ? FCS_NIBBLES - 'd1 : PAD_SIZE_NIBBLES - 'd1;
                        state       <= (padding_nibbles == 'd0) ? TX_FCS : TX_PAD;
                    end
                end             
            end
            TX_PAD: begin
                if(clk_flag_tx) begin
                    ETH_TXD <= '0;
                    if (cnt_flagTX != 'd0) begin
                        fcs         <= compute_crc(fcs, 'd0);
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        state       <= state;
                    end else begin
                        fcs         <= swap_nibbles(compute_crc(fcs, 'd0));
                        cnt_flagTX  <= FCS_NIBBLES - 'd1;
                        state       <= TX_FCS;
                    end
                end                 
            end
            TX_FCS: begin
                // Get the current nibble and take the one's complement, then
                // reverse the order of the bits, then send the new nibble.
                if(clk_flag_tx) begin
                    ETH_TXD <= {<<bit{~`SLICE(fcs, cnt_flagTX)}};
                    if (cnt_flagTX != 0) begin
                        cnt_flagTX  <= cnt_flagTX - 'd1;
                        state       <= state;
                    end else begin
                        cnt_flagTX  <= 'd0;
                        state       <= TX_GAP;
                    end
                end
            end
            // Wait the appropriate time for the Ethernet interframe gap
            TX_GAP: begin
                if(clk_flag_tx) begin
                    ETH_TXD  <= 1'd0;
                    ETH_enTX <= 1'd0;
                    if (cnt_flagTX != GAP_NIBBLES) begin
                        cnt_flagTX  <= cnt_flagTX + 'd1;
                        TX_BUSY    <= TX_BUSY;
                        state       <= state;
                    end else begin
                        cnt_flagTX  <= 'd0;
                        TX_BUSY    <= 1'd0;
                        state       <= TX_IDLE;
                    end
                end
            end
            endcase     
        end
    end
endmodule