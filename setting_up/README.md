# FPV USB Receiver – Manual Test & Debug Guide


Make sure:
- The USB receiver is plugged into your laptop.
- The FPV camera is correctly connected to the drone.
- The camera is mounted facing forward.
- Wiring is correct:
    - **Black (GND)** → first pin (left)
    - **Red (VCC)** → second pin (right)
- When the drone is powered on, you should see:
    - **Two green LEDs**
    - **One blue LED**
- This confirms that the camera module is powered correctly.

## Identify the Correct Video Device

If you don’t have `v4l2-ctl`, install it:
```
sudo apt install v4l-utils
```

**List all video devices**
```
v4l2-ctl --list-devices
```

Example output:

```
USB2.0 PC CAMERA (usb-0000:00:14.0-6):
    /dev/video1
    /dev/video4
```

The one with:
```
Video Capture
Streaming
```

You can confirm which device belongs to the receiver by:
1. Running `v4l2-ctl --list-devices`
2. Unplugging the receiver
3. Running the command again
4. Plugging it back in
5. Running the command once more
The device that appears/disappears is your receiver.

## Testing visual 


Once you know your device (for example `/dev/video1`), test the video feed.
If, for any reason, you see only static:
- Hold the button on the receiver for a few seconds.
- It will start scanning for the correct frequency.
- Wait until it locks onto the drone’s transmission.
If hardware and software are correctly configured, you should see the live camera feed once scanning finishes.
If you still see only static:
- Try a different receiver module.
- Try a different FPV camera.
- It is possible that a receiver fails to properly demodulate video even if signal strength shows 100%.


### Test With VLC

if you don't have it install:
```
sudo apt update
sudo apt install vlc
```

⚠️ Make sure you install VLC via apt, not Snap. You can verify with:
```
which vlc
```
which should return `/usr/bin/vlc`.


If you previously used Snap, clear cached paths:
```
hash -r
```

Also ensure you are in the `video` group to avoid permission issues:
```
groups
```
If video is missing:
```
sudo usermod -aG video $USER
```
Then log out and log back in.


Basic test
```
vlc v4l2:///dev/video1
```
Replace `/dev/video1` with your actual device.


### Test With guvcview

`guvcview` is often more reliable than VLC for debugging.

If not installed:
```
sudo apt install guvcview
```

Run:
```
guvcview
```
Select the correct `/dev/videoX`.

### Test Raw Stream With ffplay

Lightweight and useful for debugging:
```
ffplay /dev/video1
```



## Expected Behavior Summary

If everything is working:
- Receiver shows correct frequency
- Signal strength increases when drone is nearby
- Live image appears in VLC / guvcview / ffplay
If you see:
- 100% signal + pure static → receiver may be faulty
- Device not listed → USB issue
- Metadata-only node → wrong `/dev/videoX`
- Works on another PC but not yours → driver or VLC issue