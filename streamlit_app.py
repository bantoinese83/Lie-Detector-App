import random
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from human_subject import HumanSubject

# Load the trained model
model = joblib.load('best_rf_model.pkl')


def main():
    st.title('Lie Detector App 🕵️‍♀️ - Enhanced Police Interrogation Simulation')

    # Initialize the HumanSubject instance (Customize as needed)
    human_subject = HumanSubject(update_interval=1, lie_probability=0.4, stress_sensitivity=1.2, baseline_noise=0.2)

    # Interactive Controls in Sidebar
    st.sidebar.header('Simulation Controls ⚙️')
    lie_probability = st.sidebar.slider('Lie Probability', min_value=0.0, max_value=1.0, value=human_subject.prob_lie,
                                        step=0.05)
    stress_sensitivity = st.sidebar.slider('Stress Sensitivity', min_value=0.0, max_value=2.0,
                                           value=human_subject.stress_sensitivity, step=0.1)
    baseline_noise = st.sidebar.slider('Baseline Noise', min_value=0.0, max_value=0.5,
                                       value=human_subject.baseline_noise, step=0.05)

    # Update HumanSubject instance with interactive controls
    human_subject.prob_lie = lie_probability
    human_subject.stress_sensitivity = stress_sensitivity
    human_subject.baseline_noise = baseline_noise

    st.header('Real-time Physiological Data 📊')

    # Create placeholders for charts
    hr_chart = st.empty()
    hrv_chart = st.empty()
    eda_chart = st.empty()

    # Initialize lists to store data points for each physiological feature
    hr_data = [human_subject.features['heart_rate']]
    hrv_data = [human_subject.features['hrv']]
    eda_data = [human_subject.features['eda']]

    # Display the initial charts
    hr_chart.line_chart(hr_data, use_container_width=True)
    hrv_chart.line_chart(hrv_data, use_container_width=True)
    eda_chart.line_chart(eda_data, use_container_width=True)

    st.header('Interrogation Simulation 👮‍♀️')

    # Use the questions_answers list from HumanSubject instance
    questions_answers = human_subject.questions_answers
    random.shuffle(questions_answers)  # Shuffle questions for each session

    # Function to update charts with new data
    def update_charts(current_features):
        # Append new data points
        hr_data.append(current_features[0])
        hrv_data.append(current_features[1])
        eda_data.append(current_features[2])

        # Update the charts with the accumulated data
        hr_chart.line_chart(hr_data, use_container_width=True)
        hrv_chart.line_chart(hrv_data, use_container_width=True)
        eda_chart.line_chart(eda_data, use_container_width=True)

    # Predict the probability of lying using the trained model
    def predict_lie(features):
        prob_lie = model.predict_proba([features])[0][1]
        return prob_lie

    # Interactive questioning
    if st.sidebar.button('Start Simulation 🎬'):
        st.sidebar.write('Simulation in progress...')
        for question, is_truth in questions_answers:
            st.write(f"Question: {question}")

            # Simulate response based on the truthfulness of the answer
            is_lying = not is_truth
            human_subject.simulate_response(is_lying=is_lying)
            current_features = human_subject.get_features()
            prob_lie = predict_lie(current_features)

            # Improved Display using st.metric (Rounded off values and units)
            st.metric(label='Lie Probability 📊', value=f"{prob_lie:.2f}",
                      delta=f"{prob_lie:.2f}" if prob_lie > 0.5 else f"{1 - prob_lie:.2f}",
                      delta_color='inverse' if prob_lie > 0.5 else 'normal')
            st.metric(label='Truth/Lie 🧐', value='Truth' if not is_lying else 'Lie')
            st.metric(label='Heart Rate ❤️', value=f"{round(current_features[0])} bpm")
            st.metric(label='HRV 📈', value=f"{round(current_features[1])} ms")
            st.metric(label='EDA 💦', value=f"{current_features[2]:.2f} μS")

            # Update charts immediately with new data
            update_charts(current_features)

            # Simulate a delay for answering the question (adjust as needed)
            time.sleep(1)

        # Session summary (can be expanded with more detailed analysis)
        st.subheader('Session Summary 📄')
        st.write(f"Average Heart Rate: {round(np.mean(hr_data))} bpm")
        st.write(f"Average HRV: {round(np.mean(hrv_data))} ms")
        st.write(f"Average EDA: {round(np.mean(eda_data), 2)} μS")

        st.write('Interrogation completed. Review the questions and responses for insights.')

    # Additional visualizations and controls
    st.sidebar.subheader('Additional Visualizations 📈')
    display_histogram = st.sidebar.checkbox('Display Histograms')
    if display_histogram:
        st.subheader('Histograms of Physiological Data 📊')
        hist_bins = st.slider('Number of Bins', min_value=5, max_value=20, value=10)
        st.pyplot(draw_histograms(hr_data, hrv_data, eda_data, hist_bins))


def draw_histograms(hr_data, hrv_data, eda_data, bins):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].hist(hr_data, bins=bins, alpha=0.75, color='blue', edgecolor='black')
    axs[0].set_title('Heart Rate Histogram')
    axs[0].set_xlabel('Heart Rate (bpm)')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(hrv_data, bins=bins, alpha=0.75, color='green', edgecolor='black')
    axs[1].set_title('Heart Rate Variability Histogram')
    axs[1].set_xlabel('HRV (ms)')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(eda_data, bins=bins, alpha=0.75, color='orange', edgecolor='black')
    axs[2].set_title('Electrodermal Activity Histogram')
    axs[2].set_xlabel('EDA (μS)')
    axs[2].set_ylabel('Frequency')

    plt.tight_layout()
    return fig


# Run the app
if __name__ == '__main__':
    main()
