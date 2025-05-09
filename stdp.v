// STDP突触模块 - 实现脉冲时序依赖可塑性
module stdp_synapse #(
    parameter WEIGHT_WIDTH = 8,
    parameter TAU_P = 20,              // 长时程增强时间常数
    parameter TAU_D = 20,              // 长时程抑制时间常数
    parameter A_PLUS = 8'd5,           // 长时程增强幅度
    parameter A_MINUS = 8'd3,          // 长时程抑制幅度
    parameter W_MAX = 8'd255,          // 最大权重
    parameter W_MIN = 8'd0             // 最小权重
)(
    input wire clk,
    input wire rst_n,
    input wire learning_enable,         // 学习使能
    input wire pre_spike,               // 前神经元脉冲
    input wire post_spike,              // 后神经元脉冲
    output reg [WEIGHT_WIDTH-1:0] weight // 突触权重
);

    // 指数衰减追踪变量
    reg [15:0] pre_trace;  // 前脉冲痕迹
    reg [15:0] post_trace; // 后脉冲痕迹
    
    // 衰减系数(固定点表示)
    localparam PRE_DECAY = (1 << 10) / TAU_P;
    localparam POST_DECAY = (1 << 10) / TAU_D;
    
    // 内部权重变化
    reg [15:0] weight_change;
    
    // 前脉冲痕迹更新
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pre_trace <= 0;
        end else if (learning_enable) begin
            if (pre_spike)
                pre_trace <= pre_trace + (1 << 10); // 加固定点1.0
            else
                pre_trace <= pre_trace - ((pre_trace * PRE_DECAY) >> 10); // 指数衰减
        end
    end
    
    // 后脉冲痕迹更新
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            post_trace <= 0;
        end else if (learning_enable) begin
            if (post_spike)
                post_trace <= post_trace + (1 << 10); // 加固定点1.0
            else
                post_trace <= post_trace - ((post_trace * POST_DECAY) >> 10); // 指数衰减
        end
    end
    
    // 权重更新 - STDP核心算法
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight <= W_MAX >> 1; // 初始化为中间值
            weight_change <= 0;
        end else if (learning_enable) begin
            // 计算权重变化
            if (pre_spike) begin
                // 前脉冲发生，检查后脉冲痕迹
                weight_change <= (post_trace * A_PLUS) >> 10; // 长时程增强
            end else if (post_spike) begin
                // 后脉冲发生，检查前脉冲痕迹
                weight_change <= (pre_trace * A_MINUS) >> 10; // 长时程抑制
            end else begin
                weight_change <= 0;
            end
            
            // 应用权重变化
            if (weight_change != 0) begin
                if (pre_spike && (weight + weight_change <= W_MAX))
                    weight <= weight + weight_change;
                else if (post_spike && (weight >= weight_change))
                    weight <= weight - weight_change;
                else if (post_spike)
                    weight <= W_MIN;
            end
        end
    end
    
endmodule