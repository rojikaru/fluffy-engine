import streamlit as st

def main():
    st.title("Main Page Title")
    st.write("Hello!")

    # Input text box
    user_input = st.text_input("Enter some text:")

    # Button to submit input
    if st.button("Submit") and user_input:
        st.write(f"You entered: {user_input}")



if __name__ == "__main__":
    main()
