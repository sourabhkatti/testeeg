import time
import pyautogui

sample_count = 1
cx = 0
cy = 0
ncx = 0
ncy = 0
sw, sh = pyautogui.size()
trigger = False

while not trigger:
    cx, cy = pyautogui.position()
    if cx > 3680:
        trigger = True
    time.sleep(0.01)

print("*********** Started recording **********")
t2 = time.time()
with open('mouse_move_delta.log', 'w') as flog:
    while 1:

        t1 = time.time()
        cx, cy = pyautogui.position()
        deltax = (ncx - cx) / sw
        deltay = (ncy - cy) / sh
        tdiff = t1 - t2

        flog.write(str(sample_count) + ', ' + str(time.time()) + ", " + str(tdiff) + ", " + str(cx) + ", " + str(
            cy) + ", " + str(deltax) + ", " + str(deltay) + "\n")
        print(str(sample_count) + ', ' + str(time.time()) + ", " + str(tdiff) + ", " + str(cx) + ", " + str(
                cy) + ", " + str(deltax) + ", " + str(deltay))

        ncx = cx
        ncy = cy
        sample_count += 1
        time.sleep(1 / 128.0)
        t2 = t1

# pyautogui.moveTo(1400, 40)
# pyautogui.doubleClick()
# pyautogui.click()
# pyautogui.typewrite("www.google.com")
# pyautogui.press('enter')
