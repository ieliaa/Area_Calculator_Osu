import time
from os import _exit
from PIL import Image, ImageDraw, ImageColor
from typing import Annotated
import numpy as np
from pynput.mouse import Listener
import typer
from rich.progress import track
from rich import print as rprint

SAMPLE_RATE = 0.01

def record_movements(
    duration: int,
    grace: int,
    offset: tuple[int,int]
) -> tuple[np.ndarray[np.uint16], np.ndarray[np.uint16]]:
    """Records cursor movements for the given duration"""
    for _ in track(
        range(grace * 100),
        description=f"Waiting for {grace} seconds before recording...",
    ):
        time.sleep(0.01)

    x_input = np.array([], dtype=np.uint16)
    y_input = np.array([], dtype=np.uint16)

    def on_move(x: int, y: int) -> None:
        x -= offset[0]
        y -= offset[1]
        """Records cursor movements"""
        nonlocal x_input, y_input
        x_input = np.append(x_input, x)
        y_input = np.append(y_input, y)

    with Listener(on_move=on_move):
        print(f"Recording started for {duration} seconds...")
        # Sampling every 10ms
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < duration:
            time.sleep(SAMPLE_RATE)

    return x_input, y_input


def find_peak_near_extremes(
    values: np.ndarray[np.uint16],
    min_val: np.uint16,
    max_val: np.uint16,
    threshold_percentage: int = 5,
) -> tuple[int, int]:
    """Finds the most used point near the detected min/max values"""
    threshold_range = (max_val - min_val) * (threshold_percentage / 100)
    near_min = values[values <= min_val + threshold_range]
    near_max = values[values >= max_val - threshold_range]

    # Remove negative values
    near_min = near_min[near_min >= 0]
    near_max = near_max[near_max >= 0]

    if len(near_min) > 0:
        min_peak = np.bincount(near_min.astype(int)).argmax()
    else:
        min_peak = int(min_val)

    if len(near_max) > 0:
        max_peak = np.bincount(near_max.astype(int)).argmax()
    else:
        max_peak = int(max_val)

    return min_peak, max_peak

def draw_image(
    x_input: np.ndarray[np.uint16],
    y_input: np.ndarray[np.uint16],
    screen_width_px: int,
    screen_height_px: int,
    x_distance_px: int,
    y_distance_px: int,
    x_mean: int,
    y_mean : int
):
    # account for any offset in coordinate system
    x_max = np.max(x_input)
    y_max = np.max(y_input)
    screen_width_px = max(screen_width_px, x_max, x_mean+x_distance_px/2)
    screen_height_px = max(screen_height_px, y_max, y_mean+y_distance_px/2)

    img = Image.new("HSV",(screen_width_px, screen_height_px))
    hue = 0
    color = (int(hue),360,360)
    color_increment = 240/len(x_input)
    for i in range(len(x_input)-1):
        shape = [(x_input[i],y_input[i]),(x_input[i+1],y_input[i+1])]
        drawer = ImageDraw.Draw(img)
        drawer.line(shape, color, width=3)

        hue += color_increment
        color = (int(hue),360,360)


    top_left = (x_mean-x_distance_px/2, y_mean-y_distance_px/2)
    top_right = (x_mean+x_distance_px/2, y_mean-y_distance_px/2)
    bottom_left = (x_mean-x_distance_px/2, y_mean+y_distance_px/2)
    bottom_right = (x_mean+x_distance_px/2, y_mean+y_distance_px/2)
    shape = [top_left,top_right,bottom_right,bottom_left,top_left]
    drawer.line(shape, fill="white", width=4)

    img.show()

def write_to_file(
    x_input: np.ndarray[np.uint16],
    y_input: np.ndarray[np.uint16],
    filename: str
):
    f = open(filename,"a")
    for i in range(len(x_input)):
        f.write(f"{x_input[i]} {y_input[i]}\n")


def analyze_data(
    x_input: np.ndarray[np.uint16],
    y_input: np.ndarray[np.uint16],
    tablet_width_mm: int,
    tablet_height_mm: int,
    innergameplay_width_px: int,
    innergameplay_height_px: int,
):
    """Analyzes the movement data and finds dimensions & peak points"""
    # Get's the values in +- 3 standard deviations from the mean
    x_mean = np.mean(x_input)
    x_std_deviation = np.std(x_input)
    y_mean = np.mean(y_input)
    y_std_deviation = np.std(y_input)

    x_filtered = x_input[
        (x_input > x_mean - 3 * x_std_deviation)
        & (x_input < x_mean + 3 * x_std_deviation)
    ]
    y_filtered = y_input[
        (y_input > y_mean - 3 * y_std_deviation)
        & (y_input < y_mean + 3 * y_std_deviation)
    ]
    x_max = np.max(x_filtered)
    x_min = np.min(x_filtered)

    y_max = np.max(y_filtered)
    y_min = np.min(y_filtered)

    # Find peak usage near the filtered extremes
    x_min_peak, x_max_peak = find_peak_near_extremes(
        values=x_filtered, min_val=x_min, max_val=x_max
    )
    y_min_peak, y_max_peak = find_peak_near_extremes(
        values=y_filtered, min_val=y_min, max_val=y_max
    )

    x_distance_px = x_max_peak - x_min_peak
    y_distance_px = y_max_peak - y_min_peak
    x_distance_mm = (x_distance_px * tablet_width_mm) / innergameplay_width_px
    y_distance_mm = (y_distance_px * tablet_height_mm) / innergameplay_height_px

    rprint("\n==== RESULTS ====")
    rprint(
        "Area calculated with most used points near extremes (removed soft outliers):"
        f" [green]{x_distance_mm:.2f} x {y_distance_mm:.2f} mm [/green]"
    )
    rprint("===================")
    return x_distance_px, y_distance_px, x_mean, y_mean

def main(
    screen_width_px: int,
    screen_height_px: int,
    tablet_width_mm: float,
    tablet_height_mm: float,
    duration: int,
    offset: tuple[int,int] = (0,0),
    image: bool = False,
    file: str = "",
    grace: int = 5.0
):
    innergameplay_height_px = int((864 / 1080) * screen_height_px)
    innergameplay_width_px = int((1152 / 1920) * screen_width_px)
    typer.confirm(
        "Press Enter to start recording",
        default=True,
        show_default=False,
        prompt_suffix=" ",
    )



    x_input, y_input = record_movements(duration,grace,offset)
    x_distance_px, y_distance_px, x_mean, y_mean = analyze_data(
        x_input=x_input,
        y_input=y_input,
        tablet_width_mm=tablet_width_mm,
        tablet_height_mm=tablet_height_mm,
        innergameplay_width_px=innergameplay_width_px,
        innergameplay_height_px=innergameplay_height_px,
    )
    if image:
        draw_image(
            x_input=x_input,
            y_input=y_input,
            screen_width_px=screen_width_px,
            screen_height_px=screen_height_px,
            x_distance_px=x_distance_px,
            y_distance_px=y_distance_px,
            x_mean=x_mean,
            y_mean = y_mean
        )
    if file:
        write_to_file(
                x_input=x_input,
                y_input=y_input,
                filename=file
        )

    again = typer.confirm("Want to record again?", default=True, prompt_suffix=" ")
    if again:
        return main(
            screen_width_px,
            screen_height_px,
            tablet_width_mm,
            tablet_height_mm,
            duration,
            offset,
            image,
            file,
        )
    rprint("===================")
    rprint("Thank you for using the Area Calculator!")
    rprint(
        "If you find any issues, feel free to report it on"
        " [link=https://github.com/denwii/Area_Calculator_Osu]GitHub[/link]!"
    )
    raise typer.Exit()


if __name__ == "__main__":
    typer.run(main)
