# hybrid-music-recommendation-system
📄 Description

This project presents a hybrid architecture that combines Facial Emotion Recognition (FER) and Hand Gesture Control to deliver an emotionally adaptive, touch-free music experience. Unlike traditional recommendation systems that rely solely on listening history or user preferences, this system detects the user's real-time emotional state via a CNN-based facial analysis model and recommends music accordingly — while allowing seamless playback control through natural hand gestures tracked via MediaPipe's 21-landmark hand model.

✨ Key Features

🧠 Emotion-Driven Recommendations — CNN classifies facial expressions into emotion categories to recommend contextually relevant music in real time
🖐️ Gesture-Based Playback Control — Play, pause, skip, and adjust volume using natural hand gestures — no touch required
⚡ Low-Latency, On-Device Processing — Runs locally to minimize delay and protect sensitive biometric data
🔒 Privacy-Preserving Design — No cloud dependency for facial or gesture data
🌗 Robust Preprocessing — Handles varying lighting conditions and user posture changes for reliable performance
♿ Enhanced Accessibility — Designed for hands-free and touch-free interaction


🛠️ Tech Stack
ComponentTechnologyEmotion RecognitionConvolutional Neural Network (CNN)Gesture TrackingMediaPipe (21 Hand Landmarks)Music RecommendationHybrid Filtering (Content + Collaborative)Vision ProcessingOpenCVLanguagePython

📊 Evaluation Metrics

Recommendation Relevance
Gesture Recognition Accuracy
End-to-End System Latency


📚 Research
This system is documented in an IEEE-format research paper:
"A Hybrid Architecture for Music Recommendation and Control: Integrating Facial Emotion Recognition and Hand Gesture Interaction"
Nirmal Kumar Thirumal et al. — Department of Computer Science Engineering, AMET University, Chennai, India

Feel free to paste this directly into your GitHub repo's README.md. Let me know if you'd like badges, a setup/installation section, or a shorter one-liner description for the repo's About field!
