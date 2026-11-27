The Raspberry interface with the ADXL355 using the IIO drivers.
In a default installation all drivers should already be installed.
Otherwise, the drivers for the SPI (`adxl355_spi.c`) must be installed manually.
See ... for the installation of the SPI drivers.

If all drivers are present, raw data from the accelerometer are accessible reading the files `/sys/bus/iio/devices/iio:device0`:

 - `in_accel_x_raw`
 - `in_accel_y_raw`
 - `in_accel_z_raw`

The file `in_accel_scale` gives the scale of the raw acceleration output, in multiples of *g*.

# Configure the ADXL355 interface

## ADXL overlay

In `/boot/firmware/overlays/adxl355-spi-overlay.dts`:
```
/dts-v1/;
/plugin/;

/ {
    compatible = "brcm,bcm2835";

    fragment@0 {
        target = <&spi0>;
        __overlay__ {
            status = "okay";
            #address-cells = <1>;
            #size-cells = <0>;

            adxl355@0 {
                compatible = "adi,adxl355";
                reg = <0>;
                spi-max-frequency = <5000000>;
                interrupt-parent = <&gpio>;
                interrupts = <27 1>;        /* 1 = IRQ_TYPE_EDGE_RISING */
                interrupt-names = "DRDY";
            };
        };
    };
};
```
Then compile with `sudo dtc -@ -I dts -O dtb -o /boot/firmware/overlays/adxl355-spi-overlay.dtbo /boot/firmware/overlays/adxl355-spi-overlay.dts`

## DRDY trigger

The best way to read accelerometer data is to set a trigger, which sends a signal to the DRDY pin and tells the raspberry to hold the data into a buffer and read it every 4.096 seconds.

```
cd /sys/bus/iio/devices/iio:device0

# 0. Disable buffer and set time to real time
echo 0 | sudo tee buffer/enable
echo 100 | sudo tee buffer/watermark
echo realtime | sudo tee current_timestamp_clock

# 1. Attach the trigger
trigname=$(cat ../trigger0/name)
echo $trigname | sudo tee trigger/current_trigger
cat trigger/current_trigger     # check if correct

# 2. Enable the channels you want in the buffer
echo 1 | sudo tee scan_elements/in_accel_{x,y,z}_en
echo 1 | sudo tee scan_elements/in_timestamp_en

# 3. Sampling frequency
echo 1000 | sudo tee in_accel_sampling_frequency

# 4. Size the buffer and start capturing
echo 4096 | sudo tee buffer/length
echo 1 | sudo tee buffer/enable

# 5. High pass filter to remove DC
echo 2.47 | sudo tee in_accel_filter_high_pass_3db_frequency

# 6. Check sampling frequency
a=$(awk '/adxl355_irq/{print $2+$3+$4+$5}' /proc/interrupts)
sleep 10
b=$(awk '/adxl355_irq/{print $2+$3+$4+$5}' /proc/interrupts)
echo $(( (b-a)/10 )) Hz     # Should be ~ 1,000
```

