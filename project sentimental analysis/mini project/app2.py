import streamlit as st
import os
import sys
from tensorflow.keras.models import model_from_json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model_json_path = "D:/mini project/model.json"


def load_model(model_json_path, model_weights_path, tokenizer_path):
    if not os.path.exists(model_json_path):
        st.error(f"Error: {model_json_path} not found")
        st.stop()

    if not os.path.exists(model_weights_path):
        st.error(f"Error: {model_weights_path} not found")
        st.stop()

    if not os.path.exists(tokenizer_path):
        st.error(f"Error: {tokenizer_path} not found")
        st.stop()

    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(model_weights_path)

    with open(tokenizer_path, "rb") as tokenizer_file:
        loaded_tokenizer, maxlen = pickle.load(tokenizer_file)

    return loaded_model, loaded_tokenizer, maxlen

def analyze_sentiment(review, loaded_model, loaded_tokenizer, maxlen):
    review_sequence = loaded_tokenizer.texts_to_sequences([review])
    review_padded = pad_sequences(review_sequence, maxlen=maxlen)
    result = loaded_model.predict(review_padded)[0]
    sentiment = np.argmax(result)
    if sentiment == 0:
        return "Negative"
    elif sentiment == 1:
        return "Neutral"
    else:
        return "Positive"

def main():
    st.title("WNS Watch Company Dashboard")

    # Sidebar with navigation buttons
    page = st.sidebar.radio("Navigation", ["Home", "Model", "About Us"])

    if page == "Home":
        st.header("Welcome to WNS Watch Company")
        st.write("Browse our collection of watches below:")

        watch_details = {
            "Billionare Timeless": {
                "image_url": "https://luxebook.in/wp-content/uploads/2023/04/jacobandcocom_411929602.webp",
                "story": "At Watches & Wonders 2023 in Geneva, Jacob & Co. introduced their newest flagship piece, calling it the Billionaire Timeless Treasure with a price tag of 20 million dollars. Which is something completely different than they’ve ever showed. The piece is covered with 425 stones in perfect color (fancy yellow and fancy intense yellow) thanks to the presence of traces of nitrogen within their carbon structure. All these stones making the watch look as one piece, which must be a heavy one to wearJacob and Co. didn’t just produce this piece overnight, with the company searching for the perfect gems for 3.5 years already. They’ve gathered them, where each one was scrutinized at rough stage, at cut stage, before and after setting. Jacob & Co. really created a unique piece, which is extremely exclusive and with high-quality diamonds. "
            },
            "Richard-Mille": {
                "image_url": "https://media.richardmille.com/wp-content/uploads/2020/01/19101812/richard-mille-rm-002-v2-tourbillon-45269.jpg?dpr=1&width=2000",
                "story": "Manual winding tourbillon movement with hours, minutes, function selector, power-reserve and torque indicators.Giving existing materials a new impulse by applying them in a new manner, as well as the implementation of new materials to watchmaking, is completely second nature to Richard Mille Watches’ philosophy.In 2005, for the first time in watchmaking history and after years of research and development, a movement baseplate in carbon nanofiber were created and tested in the RM 006.The technical results were so remarkable that Richard Mille therefore decided to expand the use of carbon nanofiber further in its tourbillon range, opening up a new era in watchmaking. The V2 version of the RM 002-V1 is released."
             },
            "Patek Philippe": {
                "image_url": "https://static.patek.com/images/articles/face_white/350/7118_1453G_001_1.jpg",
                "story": "Patek Philippe offers a new alliance between elegant sports design and Haute Joaillerie in a ladies’ Nautilus watch adorned with 1,500 brilliant-cut diamonds (6.53 cts) and 876 emeralds (3.68 cts), with a snow-set case, bezel and bracelet. The emerald-set bezel and inner links of the bracelet magnify the inimitable geometry of the Nautilus.The white gold dial is distinguished by rows of brilliant-cut diamonds set in the wave pattern typical of the ladies’ Nautilus collection.Self-winding 26-330 S caliber is visible through a transparent sapphire case-back.The bracelet features the patented Patek Philippe fold-over clasp secured by four independent catches."
            },
            "Astronomia Tourbillon Dragon": {
                "image_url": "https://i.pinimg.com/736x/76/b5/d1/76b5d1f790bc71aa7ba215235ceda93d.jpg",
                "story": "Astronomia is about the display of extraordinary, miniature, finely crafted objects. Jacob & Co. has made it a platform to express the entire span of its creativity. Astronomia Dragon is a subcollection of its own standing. It gathers all the know-how Jacob & Co. accumulated in watchmaking, high jewelry and artistic crafts and applies it to Jacob & Co.'s most emblematic, striking and powerful creature : the dragon. Mythical and very much alive, positive and mighty, the dragon is by far the most revered creature in the watchmaking playbook. Jacob & Co. has been manufacturing dragon-themed timepieces for ten years, without interruption, always at the highest level of craft, symbolism and exclusivity. Astronomia Dragon features a majestic, three-dimensional, expressive and ominous wingless dragon inspired by the Asian tradition. The rose gold sculpture has been the focus of several timepieces, all of which are unique pieces. The dragon is either left in its natural, solid gold state, or hand-painted. The entire creature is highly expressive and changes as iterations go. One thing remains : the open mouth, the potent claws, the coiled body, the detailed scales all spell out excellence in craft."
            }
        }

        for watch_name, details in watch_details.items():
            st.subheader(watch_name)
            st.image(details["image_url"], caption=watch_name, use_column_width=True)
            st.write(details["story"])

    elif page == "Model":
        st.header("Model Page")
        st.write("Select a watch to view details:")
        
        # Display images as buttons
        watch_images = {
            "Billionare Timeless": "https://luxebook.in/wp-content/uploads/2023/04/jacobandcocom_411929602.webp",
            "Richard-Mille": "https://media.richardmille.com/wp-content/uploads/2020/01/19101812/richard-mille-rm-002-v2-tourbillon-45269.jpg?dpr=1&width=2000",
            "Patek Philippe": "https://static.patek.com/images/articles/face_white/350/7118_1453G_001_1.jpg",
            "Astronomia Tourbillon Dragon": "https://i.pinimg.com/736x/76/b5/d1/76b5d1f790bc71aa7ba215235ceda93d.jpg"
        }
        selected_watch = st.selectbox("Select a watch:", list(watch_images.keys()))
        selected_watch_image_url = watch_images[selected_watch]
        
        if st.button(selected_watch):
            st.image(selected_watch_image_url, caption=selected_watch, use_column_width=True)
            st.button("Buy")
            st.button("Add to Cart")
            
        review = st.text_input("Enter your Rating Out Of 5:")
        summary = st.text_input("Enter the Summary: ")
        
        st.write("Reviews:")
        st.write("Enter your review below:")
        review = st.text_area("Review", "")
        if st.button("Analyze Sentiment"):
            loaded_model, loaded_tokenizer, maxlen = load_model(model_json_path, "D:/mini project/model_weights.weights.h5", "D:/mini project/tokenizer.pickle")
            sentiment = analyze_sentiment(review, loaded_model, loaded_tokenizer, maxlen)
            st.write("Sentiment:", sentiment)

    elif page == "About Us":
        st.header("About Us Page")
        st.image("https://c8.alamy.com/comp/2RDY5MB/wns-triangle-letter-logo-design-with-triangle-shape-wns-triangle-logo-design-monogram-wns-triangle-vector-logo-template-with-red-color-wns-triangul-2RDY5MB.jpg", caption="WNS Watch Company Logo")
        st.write("WNS Watch Company is a leading watch manufacturer known for its innovative designs and high-quality timepieces. With a passion for craftsmanship and attention to detail, we create watches that blend style and functionality.")
        st.write("For more information, visit our website at [www.wnswatches.com](www.wnswatches.com)")

if __name__ == "__main__":
    main()
