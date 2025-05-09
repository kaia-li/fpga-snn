// 改进的漏电积分发放(LIF)神经元模型 - 细粒度并行计算
module neuron_lif_improved #(
    parameter THRESHOLD = 16'd550,      // 阈值电压 (x10精度)
    parameter REST_POTENTIAL = 16'd650, // 静息电位 (x10精度)
    parameter RESET_POTENTIAL = 16'd700, // 重置电位 (x10精度)
    parameter TAU_M = 8'd20,            // 膜时间常数
    parameter REFRACTORY_PERIOD = 4'd2  // 不应期(时间步数)
)(
    input wire clk,
    input wire rst_n,
    input wire enable,                  // 使能信号
    input wire [15:0] input_current,    // 输入电流
    output reg [15:0] membrane_potential, // 膜电位
    output reg spike                    // 脉冲输出
);
    
    // 固定点运算的常数
    localparam FIXED_POINT_SHIFT = 10;
    localparam TAU_FACTOR = (1 << FIXED_POINT_SHIFT) / TAU_M;
    
    // 不应期计数器
    reg [3:0] refractory_counter;
    
    // 内部信号
    reg [15:0] leak_current;
    
    // 计算漏电电流 - 固定点运算提高精度
    always @(*) begin
        if (membrane_potential > REST_POTENTIAL)
            leak_current = ((membrane_potential - REST_POTENTIAL) * TAU_FACTOR) >> FIXED_POINT_SHIFT;
        else if (membrane_potential < REST_POTENTIAL)
            leak_current = ((REST_POTENTIAL - membrane_potential) * TAU_FACTOR) >> FIXED_POINT_SHIFT;
        else
            leak_current = 0;
    end
    
    // 膜电位更新 - 主计算过程
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential <= REST_POTENTIAL;
            spike <= 0;
            refractory_counter <= 0;
        end else if (enable) begin
            if (refractory_counter > 0) begin
                // 处于不应期
                refractory_counter <= refractory_counter - 1;
                membrane_potential <= RESET_POTENTIAL;
                spike <= 0;
            end else begin
                // 正常更新膜电位
                if (membrane_potential > REST_POTENTIAL)
                    membrane_potential <= membrane_potential - leak_current + input_current;
                else
                    membrane_potential <= membrane_potential + leak_current + input_current;
                
                // 检查是否发放
                if (membrane_potential >= THRESHOLD) begin
                    spike <= 1;
                    refractory_counter <= REFRACTORY_PERIOD;
                end else begin
                    spike <= 0;
                end
            end
        end
    end
    
endmodule