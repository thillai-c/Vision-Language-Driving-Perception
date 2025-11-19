# LinkedIn Post - Vision-Language Driving Perception

🚗 How can autonomous vehicles better understand and explain their driving decisions?

Most current systems treat perception and planning as separate modules. When a vehicle makes a decision—like changing lanes or slowing down—it's often a black box. We know *what* happened, but not *why* the AI chose that action, which is critical for safety and debugging.

Vision-Language Models offer an interesting approach: they can understand driving scenes through visual input and explain their reasoning in natural language. But adapting these models for real-world driving tasks isn't straightforward. You need custom fine-tuning pipelines, efficient training setups, and ways to work with actual driving datasets.

**What I built:**

I created Vision-Language Driving Perception—an open-source framework that fine-tunes VLMs (InternVL, InternLM2, Phi3) specifically for autonomous driving decision planning.

- Full fine-tuning pipeline with distributed training support (DeepSpeed ZeRO)
- Integration with nuPlan/Waymo datasets for real driving scenarios
- Flash Attention optimizations to make training more efficient
- TensorRT inference acceleration for real-time performance
- Evaluation framework that measures action prediction accuracy

The models learn to predict driving actions (speed control, lane changes, turns) and can explain their decisions in natural language. For example, instead of just outputting "change lane," the model might say "$$change lane to the right$$ The adjacent lane is clear, and the current lane has a slower vehicle ahead."

It's still a research project, but I think it shows promise for making autonomous driving systems more interpretable. The code is open-source, so others can build on it or adapt it for their own datasets.

Would love to hear thoughts from others working in autonomous driving or vision-language models!

🔗 GitHub: https://github.com/thillai-c/Vision-Language-Driving-Perception

#AutonomousDriving #VisionLanguageModels #AI #MachineLearning #OpenSource #ComputerVision

