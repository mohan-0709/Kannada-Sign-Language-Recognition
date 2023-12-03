# Kannada Vyanjanagalu Recognition Project

This project aims to efficiently recognize Kannada Vyanjanagalu with higher accuracy and reduced inference time using machine learning techniques.

## Abstract

Sign language serves as a means of communication for speech and hearing-impaired individuals. This project focuses on the real-time recognition of Static Kannada Sign Language, Vyanjanagalu, utilizing machine learning. The dataset comprises 3,400 images representing 34 static signs of Kannada Vyanjanagalu. Various machine learning models were explored, with the Gated Recurrent Units(GRU) and Support Vector Machine(SVM) achieving an accuracy rate above 99%. This technology aims to enhance the quality of life for the hearing-impaired population by improving communication through sign language recognition.

## Dataset

The curated dataset consists of 3,400 static images depicting 34 distinct signs of Kannada Vyanjanagalu, each with 100 images. These signs involve single-hand gestures captured at 640 x 480 pixels resolution. The dataset's validation was confirmed by the Association of People with Disabilities(APD) in Bengaluru.

## System Architecture

![System Architecture](path/to/system_architecture_image.png)

### Feature Extractor

The project utilizes Mediapipe Hands, a tool developed by Google, for extracting hand landmarks. This tool precisely identifies 21 landmarks per hand gesture under various lighting conditions, providing normalized x, y, z coordinates. The output is organized into numpy arrays, each adopting a (100, 21, 3)-dimensional matrix configuration, optimizing computational efficiency and facilitating data management.

### Models and Results

![Models and Results]
(D:\PES1UG20CS563\Sem 7\Capstone Phase - 2\KSL\CODE\MODELS\SVM\confusion_matrix.png)
(D:\PES1UG20CS563\Sem 7\Capstone Phase - 2\KSL\CODE\MODELS\GRU\average_precision.png)

Several machine learning models were evaluated, with GRU and SVM achieving an accuracy rate above 99%.

### Real-time Output

![Real-time Outputs]
(D:\PES1UG20CS563\Sem 7\Capstone Phase - 2\KSL\CODE\ಕ.png)
(D:\PES1UG20CS563\Sem 7\Capstone Phase - 2\KSL\CODE\ಫ.png)

## Conclusion

The project contributed a comprehensive dataset and explored multiple machine learning models, achieving exceptional accuracy in Kannada Vyanjanagalu recognition. Notable achievements include reduced inference time, real-time capability, adaptability to lighting conditions, and robustness against occlusions. The technology demonstrates proficiency in recognizing complex signs, accommodating multiple signers, and enabling translation to the Kannada language. The implementation of real-time recognition exemplifies practical applications, laying the groundwork for robust sign language recognition technologies.

This research aims to significantly advance technologies benefiting individuals with hearing impairments by facilitating accurate and efficient recognition of sign languages.
