## 🛣️ Driving Scene Understanding Dataset (Meta Action Format)
This dataset provides annotated urban driving scenes from the perspective of an ego vehicle. Each scene contains a front-facing image and a structured conversation that guides an AI assistant to predict the ego vehicle’s immediate future action using meta-actions.

## 📁 Dataset Structure
Each sample in the dataset is represented as a JSON object with the following fields:

```json
{
  "id": int,
  "image": str,
  "width": int,
  "height": int,
  "conversations": [
    {
      "from": "human",
      "value": "Instruction text"
    },
    {
      "from": "gpt",
      "value": "$$Meta Action$$ Scene Description"
    }
  ]
}
```


- **id**: Unique identifier for each data sample.

- **image**: Filename of the corresponding front-view image (e.g., sample_0.jpg).

- **width/height**: Resolution of the image.

- **conversations**: A structured list representing an instruction (from: "human") and the AI assistant’s response (from: "gpt"), which includes:

  - A meta-action that describes the ego car's immediate maneuver.

  - A scene description detailing the driving context and relevant surroundings.

## 🧠 Meta-Actions
The assistant must describe the ego vehicle’s immediate behavior using one or more of the following predefined meta-actions:

- **Speed-control**: speed up, slow down, stop, wait

- **Turning**: turn left, turn right, turn around

- **Lane-control**: change lane, shift slightly to the left or right

Each response must begin with a `$$Meta Action$$`, followed by a detailed natural language description of the scene.

## ✨ Example Output
```text
$$speed up$$ The scene depicts an urban road during daytime with cloudy weather. The ego vehicle is centered in its lane, approaching an intersection where it needs to proceed forward. The environment includes a traffic light and several parked cars on the side, requiring cautious acceleration.
```