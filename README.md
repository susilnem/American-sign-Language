# American Sign Language to Text

Real-time ASL gesture recognition that translates hand signs into text and speech — built with a CNN + MediaPipe hand tracking pipeline.

This is a final year project aimed at improving communication accessibility between the deaf and hearing communities.

## Output

![Output 1](./documentation/images/1.png)

![Output 2](./documentation/images/2.png)

![Output 3](./documentation/images/3.png)

![Output 4](./documentation/images/4.png)

![Output 5](./documentation/images/5.png)

## How it works

1. Webcam captures your hand in real time
2. MediaPipe extracts 21 hand landmark points
3. Landmarks are drawn as a skeleton on a blank canvas
4. A CNN classifies the skeleton into one of 8 visual groups
5. Geometry rules pick the exact letter within the group
6. Word suggestions (pyenchant) and text-to-speech (eSpeak) complete the experience

See [workflow.md](./workflow.md) for the full technical breakdown.

## Getting started

See [src/README.md](./src/README.md) for setup and run instructions.

## Model performance

![Model accuracy](./documentation/Model%20Accuracy.png)

![Confusion matrix](./documentation/Confusion%20matrix.png)
