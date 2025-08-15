# PWM Brightness Control for 7-Segment Display

## Overview
This implementation adds PWM (Pulse Width Modulation) brightness control functionality to 7-segment display controllers. The system supports 8 brightness levels (0-7) controlled via 3-bit switch inputs and operates at 1kHz PWM frequency to avoid visible flicker.

## Features
- **8 Brightness Levels**: Controlled via 3-bit switches (levels 0-7)
- **1kHz PWM Frequency**: High enough to avoid visible flicker
- **Linear Duty Cycle Mapping**: 
  - Level 0: 0% duty cycle (always off)
  - Level 1: 12.5% duty cycle
  - Level 2: 25% duty cycle
  - Level 3: 37.5% duty cycle
  - Level 4: 50% duty cycle
  - Level 5: 62.5% duty cycle
  - Level 6: 75% duty cycle
  - Level 7: 100% duty cycle (always on)
- **Maintains Existing Functionality**: All original BCD to 7-segment conversion preserved
- **Clean Integration**: Minimal changes to existing codebase

## Module Hierarchy

```
bcd2led7seg (Top Module)
├── seven_seg_controller (BCD to 7-segment conversion)
└── pwm_brightness_controller (PWM generation)
```

## Modules Description

### 1. `pwm_brightness_controller`
**Purpose**: Generates PWM signals for brightness control
- **Input Clock**: 125MHz system clock
- **PWM Frequency**: 1kHz (configurable)
- **Resolution**: 8 levels (3-bit input)
- **Features**: Reset support, linear duty cycle mapping

### 2. `bcd2led7seg`
**Purpose**: Top-level module integrating BCD conversion with PWM brightness
- **Inputs**: 
  - `bcd_input[3:0]`: BCD value (0-15, displays 0-9, blank for invalid)
  - `brightness_switches[2:0]`: Brightness level (0-7)
- **Outputs**:
  - `seg_out[7:0]`: Original 7-segment pattern
  - `seg_out_pwm[7:0]`: PWM-controlled 7-segment pattern
  - `pwm_enable`: PWM enable signal (for debugging)

### 3. `traffic_light_controller_pwm`
**Purpose**: Enhanced traffic light controller with PWM brightness
- Integrates original traffic light functionality with PWM display control
- **Additional Input**: `brightness_switches[2:0]`
- **Additional Output**: `seg_display_pwm[7:0]`

### 4. `pwm_brightness_demo`
**Purpose**: Simple demonstration module showing PWM brightness control
- Direct BCD input via switches
- Visual brightness level feedback via LEDs
- Valid BCD indication

## Implementation Details

### PWM Generation
- **Clock Domain**: 125MHz system clock
- **PWM Period**: 125,000 clock cycles (1ms at 125MHz)
- **PWM Step**: 15,625 clock cycles per brightness level
- **Counter Width**: 17 bits (log2(125,000))

### Duty Cycle Calculation
```systemverilog
duty_threshold = brightness_level * (PWM_PERIOD / 8)
pwm_out = (counter < duty_threshold) ? 1 : 0
```

### Special Cases
- **Level 0**: Always output 0 (display off)
- **Level 7**: Always output 1 (maximum brightness)
- **Reset**: PWM output forced to 0

## Testing

### Simulation Results
The testbench validates:
1. **BCD Conversion**: All digits 0-9 correctly converted
2. **PWM Duty Cycles**: Verified accurate duty cycle for each brightness level
3. **Reset Functionality**: PWM correctly responds to reset
4. **Integration**: Traffic light controller maintains functionality with added PWM

### Test Coverage
- BCD values 0-9 and invalid inputs
- All brightness levels 0-7
- Reset behavior
- Rapid brightness changes
- Long-term operation

## Usage Examples

### Basic Usage
```systemverilog
bcd2led7seg display (
    .clk(clk_125mhz),
    .reset(reset),
    .bcd_input(digit),           // 4-bit BCD input
    .brightness_switches(sw[2:0]), // 3-bit brightness control
    .seg_out_pwm(seven_seg)      // PWM-controlled output
);
```

### Traffic Light Integration
```systemverilog
traffic_light_controller_pwm traffic_controller (
    .clk(clk_125mhz),
    .rst(reset),
    .brightness_switches(brightness_sw),
    .car_light(traffic_lights),
    .seg_display_pwm(countdown_display)
);
```

## File Structure
```
rtl/
├── pwm_brightness_controller.sv   # PWM generator module
├── bcd2led7seg.sv                # Main BCD to 7-seg with PWM
├── traffic_light_controller_pwm.sv # Enhanced traffic controller
├── pwm_brightness_demo.sv         # Demo module
└── traffic_light_controller.sv    # Original modules (seven_seg_controller)

tb/
├── bcd2led7seg_tb.sv             # PWM brightness testbench
└── traffic_light_controller_pwm_tb.sv # Enhanced controller testbench
```

## Synthesis Considerations
- **Resource Usage**: Minimal additional resources (one 17-bit counter, combinational logic)
- **Timing**: PWM frequency much lower than system clock, no timing concerns
- **Power**: PWM reduces average power consumption at lower brightness levels
- **FPGA Compatibility**: Standard Verilog constructs, compatible with all major FPGA vendors

## Future Enhancements
- Non-linear brightness mapping (gamma correction)
- Variable PWM frequency control
- Multiple display support with independent brightness control
- Automatic brightness adjustment based on ambient light sensor