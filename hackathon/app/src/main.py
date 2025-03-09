from full5 import call as call_cv
import flet_audio as fta
import flet as ft
import asyncio
import warnings

warnings.filterwarnings("ignore")

links = {
    "forward": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/b1f6924e9c9888260279a9ab01d817afdb5c2290/Forward.mp3?raw=True"),
    "left": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/b1f6924e9c9888260279a9ab01d817afdb5c2290/Left.mp3?raw=True"),
    "right": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/b1f6924e9c9888260279a9ab01d817afdb5c2290/Right.mp3?raw=True"),
    "stop": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/b1f6924e9c9888260279a9ab01d817afdb5c2290/Stop.mp3?raw=True"),
    "vibration": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/fba6767b1aa7962f0403d4be11b49fd0d8136fd8/Vibration%2520Only.mp3?raw=True"),
    "voice": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/fba6767b1aa7962f0403d4be11b49fd0d8136fd8/Voice%2520only.mp3?raw=True"),
    "both": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/fba6767b1aa7962f0403d4be11b49fd0d8136fd8/Voice%2520Plus%2520Vibration.mp3?raw=True"),
    "number_prompt": fta.Audio(src=r"https://github.com/asrvn/blindsight/blob/e1dd237e5b205f262b52de5ace17efaca0d70712/Please%2520enter%2520a%25205%2520dig.mp3?raw=True")
}

notif_setting = "voice"
text_display: ft.Text = ft.Text()
hf = ft.HapticFeedback()

async def update(text: str):
    global text_display
    text_display.value = text.upper()
    text_display.update()
    if notif_setting != "vibration":
        for i in links:
            links[i].pause()

    if "left" in text:
        if notif_setting != "vibration":
            links["left"].play()
        if notif_setting != "voice":
            hf.heavy_impact()
    elif "right" in text:
        if notif_setting != "vibration":
            links["right"].play()
        if notif_setting != "voice":
            hf.heavy_impact()
            await asyncio.sleep(0.5)
            hf.heavy_impact()
    elif "forward" in text:
        if notif_setting != "vibration":
            links["forward"].play()
        if notif_setting != "voice":
            hf.vibrate()
    elif "stop" in text:
        if notif_setting != "vibration":
            links["stop"].play()
        if notif_setting != "voice":
            hf.vibrate()

async def main(page: ft.Page):
    global text_display, links
    text_display = ft.Text("", size=60)
    title = ft.Text(value="BlindSight", size=50, weight=ft.FontWeight.BOLD, font_family="Giga Sans")

    page.fonts = {
        "Open Sans": "/fonts/OpenSans-Regular.ttf",
        "Giga Sans": r"https://github.com/asrvn/blindsight/blob/c18479e10a55f274fb7746af620e7941c3a74a21/Locomotype%20%20GigaSansRegular.otf?raw=True"
    }

    for i in links:
        page.overlay.append(links[i])

    async def number_changed(e: ft.ControlEvent):
        numbers = list(filter(lambda i: i.isdigit(), e.control.value))
        if len(numbers) == 5:
            col.controls.remove(number_input)
            col.controls.remove(title)
            col.update()

            loop = asyncio.get_running_loop()
            asyncio.create_task(asyncio.to_thread(call_cv, update, loop))

    number_input = ft.TextField(
        label="Please enter a 5-digit building number",
        keyboard_type=ft.KeyboardType.NUMBER,
        autofocus=True,
        on_change=number_changed
    )

    page.overlay.append(hf)

    col = ft.Column(
        [
            ft.Image(
                src="https://github.com/asrvn/blindsight/blob/e8db30b1822fc98d93cda2949e8e4f48a8f14a9a/logo.png?raw=True",
                width=200,
                height=200
            ),
            title,
            number_input,
            ft.Container(
                text_display,
                alignment=ft.alignment.center,
            ),
        ],
        expand=True,
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=5
    )

    def shake(e):
        global notif_setting, links
        if notif_setting == "vibration":
            notif_setting = "voice"
        elif notif_setting == "voice":
            notif_setting = "both"
        elif notif_setting == "both":
            notif_setting = "vibration"
        for i in links:
            links[i].pause()
        links[notif_setting].play()

    shd = ft.ShakeDetector(
        minimum_shake_count=1,
        shake_slop_time_ms=1000,
        shake_count_reset_time_ms=1000,
        on_shake=shake
    )
    page.overlay.append(shd)

    page.add(
        ft.SafeArea(
            col,
            expand=True,
        )
    )

ft.app(main, view=ft.WEB_BROWSER)