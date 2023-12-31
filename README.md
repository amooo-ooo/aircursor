# aircursor
Aircursor is a simple, mini, python project that allows users to control their cursor with movements of their hands. Powered by open-cv, mediapipe, numpy and pyautogui. 

## Instructions
The cursor follows the centre of your hand when the program initiates. 

> [!NOTE]
> Not the palm, but the centre of your hands according to your fingers. 

Which means you are able to fine-tune the cursor position by moving your fingers. 

There are only two simple actions that are implemented in Aircursor at the moment:

| Hand Movement  | Operation |
| ------------- | ------------- |
| Closed Palm  | Mouse-Down  |
| Open Palm  | Mouse-Up  |

The cursor works normally. In order to left-click, simply close your palm and open them. This means users are able to select texts or images and drag them with intuitive hand movements. 

## Quick Start

Clone the project:
```shell
git clone https://github.com/amooo-ooo/aircursor
```

Install required dependencies:
```shell
pip install -r requirements.txt
```

Quick start the program:
```shell
python main.py
```

> [!IMPORTANT]
> Quit the program by pressing `ctrl + q`!

## Parameters
By default, no external windows open, and the program operates in the background in order to maximise performance. However, there are arguments you can pass for customisation:

| Argument  | Default | Settings |
| ------------- | ------------- |------------- |
| `--show_camera`  | `False` | Opens a seperate window displaying camera. |
| `--show_hands`  | `False` | Display user's hand as a wireframe on the external window. |
| `--sensitivity=0` | `2.0` | Adjust the sensitivity of the cursor. |
|  `--smoothness=0` | `2` | Adjust the smoothness of movements of the cursor (causes cursor delay). |
| `--resolution=0` | `640x480` | Adjust the resolution for hand model proccessing. |

For example: 
```shell
python main.py --show_camera
```

## Contact
You can contact me at [amor.budiyanto@gmail.com](mailto:amor.budiyanto@gmail.com).