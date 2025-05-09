// SNN FPGA顶层模块 - 与主机交互
module snn_fpga_top (
    input wire clk,
    input wire rst_n,
    
    // 主机接口
    input wire [31:0] host_addr,
    input wire [31:0] host_data_in,
    output reg [31:0] host_data_out,
    input wire host_write_en,
    input wire host_read_en,
    
    // 控制信号
    input wire start_compute,
    output wire compute_done,
    
    // 数据缓冲区
    input wire [7:0] pixel_buffer [0:3071],  // 输入图像缓冲
    output reg [7:0] result_buffer [0:9]     // 结果缓冲(分类结果)
);

    // 内部存储器 - 用于统计脉冲计数
    reg [15:0] spike_counts [0:9];
    
    // SNN计算核心实例化
    wire [9:0] output_spikes;
    
    snn_computation_core #(
        .INPUT_SIZE(3072),
        .HIDDEN_SIZE(500),
        .OUTPUT_SIZE(10),
        .TIME_STEPS(100)
    ) snn_core (
        .clk(clk),
        .rst_n(rst_n),
        .start_compute(start_compute),
        .input_data(pixel_buffer),
        .learning_mode(1'b1),  // 使能学习
        .output_spikes(output_spikes),
        .compute_done(compute_done)
    );
    
    // 输出脉冲计数
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 10; i = i + 1) begin
                spike_counts[i] <= 0;
                result_buffer[i] <= 0;
            end
        end else if (start_compute && !compute_done) begin
            // 统计脉冲
            for (i = 0; i < 10; i = i + 1) begin
                if (output_spikes[i])
                    spike_counts[i] <= spike_counts[i] + 1;
            end
        end else if (compute_done) begin
            // 仿真结束，将结果写入result_buffer
            for (i = 0; i < 10; i = i + 1) begin
                result_buffer[i] <= spike_counts[i][7:0];  // 取低8位作为结果
                spike_counts[i] <= 0;  // 重置计数器
            end
        end
    end
    
    // 主机接口逻辑 - 允许读写参数和控制
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            host_data_out <= 32'h0;
        end else if (host_read_en) begin
            case (host_addr[7:0])
                8'h00: host_data_out <= {22'h0, compute_done, 9'h0};
                8'h04: host_data_out <= {24'h0, result_buffer[0]};
                8'h08: host_data_out <= {24'h0, result_buffer[1]};
                8'h0C: host_data_out <= {24'h0, result_buffer[2]};
                8'h10: host_data_out <= {24'h0, result_buffer[3]};
                8'h14: host_data_out <= {24'h0, result_buffer[4]};
                8'h18: host_data_out <= {24'h0, result_buffer[5]};
                8'h1C: host_data_out <= {24'h0, result_buffer[6]};
                8'h20: host_data_out <= {24'h0, result_buffer[7]};
                8'h24: host_data_out <= {24'h0, result_buffer[8]};
                8'h28: host_data_out <= {24'h0, result_buffer[9]};
                default: host_data_out <= 32'h0;
            endcase
        end
    end
    
endmodule