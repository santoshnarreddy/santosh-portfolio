"""
Visualization utilities for the Streamlit app.
Author: Santosh Narreddy
"""

import cv2
import numpy as np
import plotly.graph_objects as go

EMOTION_COLORS_BGR = {
    'Happy':     (80, 220, 80),
    'Sad':       (220, 80, 80),
    'Angry':     (60, 60, 220),
    'Surprised': (60, 200, 240),
    'Neutral':   (170, 170, 170),
    'Fearful':   (180, 60, 180),
    'Disgusted': (60, 160, 60),
}

EMOTION_COLORS_HEX = {
    'Angry': '#ff4444', 'Disgusted': '#44aa44', 'Fearful': '#9944cc',
    'Happy': '#44cc44', 'Neutral': '#aaaaaa', 'Sad': '#4488ff', 'Surprised': '#ffaa00',
}


def draw_emotion_overlay(img_bgr, predictor, show_bbox=True, min_confidence=0.3):
    """Draw bounding boxes + labels on the image. Returns (annotated_img, detections)."""
    detections = predictor.predict(img_bgr, min_confidence=min_confidence)

    for det in detections:
        x, y, w, h = det['bbox']
        emotion = det['emotion']
        conf    = det['confidence']
        color   = EMOTION_COLORS_BGR.get(emotion, (200, 200, 200))

        if show_bbox:
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)

        label = f"{emotion}  {conf:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img_bgr, (x, y - th - 12), (x + tw + 8, y), color, -1)
        cv2.putText(img_bgr, label, (x + 4, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return img_bgr, detections


def plot_emotion_bars(all_probs: dict) -> go.Figure:
    """Return a Plotly horizontal bar chart of emotion probabilities."""
    emotions = list(all_probs.keys())
    probs    = [v * 100 for v in all_probs.values()]
    colors   = [EMOTION_COLORS_HEX.get(e, '#00c7f7') for e in emotions]

    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probs],
        textposition='auto',
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#c9d1d9',
        xaxis=dict(range=[0, 100], showgrid=False, ticksuffix='%', color='#8b949e'),
        yaxis=dict(showgrid=False, color='#c9d1d9'),
    )
    return fig
