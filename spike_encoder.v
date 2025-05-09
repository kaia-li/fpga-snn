// 脉冲编码器 - 将像素数据转换为脉冲序列
module spike_encoder #(
    parameter INPUT_SIZE = 3072,  // 32x32x3
    parameter MAX_RATE = 255,     // 最大发放率
    parameter TIME_STEPS = 100    // 时间步长
)(
    input wire clk,
    input wire rst_n,
    input wire [7:0] pixel_data [0:INPUT_SIZE-1],  // 原始像素数据
    input wire pixel_valid,                        // 像素数据有效
    input wire time_step_pulse,                    // 时间步进脉冲
    output reg [INPUT_SIZE-1:0] spike_out          // 输出脉冲（每个输入一位）
);
    
    // 存储每个输入的概率阈值
    reg [7:0] firing_thresholds [0:INPUT_SIZE-1];
    
    // 用于随机数生成的LFSR
    reg [15:0] lfsr;
    wire [7:0] random_value;
    
    // 脉冲生成计数器
    reg [7:0] time_counter;
    
    // LFSR为随机数生成器
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr <= 16'hACE1;  // 非零初始值
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
        end
    end
    
    assign random_value = lfsr[7:0];  // 取低8位作为随机数
    
    // 时间步进计数器
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            time_counter <= 0;
        end else if (time_step_pulse) begin
            if (time_counter < TIME_STEPS - 1)
                time_counter <= time_counter + 1;
            else
                time_counter <= 0;
        end
    end
    
    // 加载像素数据并转换为发放阈值
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < INPUT_SIZE; i = i + 1) begin
                firing_thresholds[i] <= 0;
            end
        end else if (pixel_valid) begin
            for (i = 0; i < INPUT_SIZE; i = i + 1) begin
                firing_thresholds[i] <= MAX_RATE - pixel_data[i];  // 反转，使较大像素值产生更多脉冲
            end
        end
    end
    
    // 基于时间依赖的泊松过程生成脉冲
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < INPUT_SIZE; i = i + 1) begin
                spike_out[i] <= 0;
            end
        end else begin
            for (i = 0; i < INPUT_SIZE; i = i + 1) begin
                // 比较随机值与当前时间步长的阈值
                // 像素值越大，随机值小于阈值的概率越小，产生脉冲的概率越大
                spike_out[i] <= (random_value < firing_thresholds[i]) ? 1'b1 : 1'b0;
            end
        end
    end
    
endmodule