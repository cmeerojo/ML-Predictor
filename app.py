import streamlit as st
import pandas as pd
import joblib
import math

model=joblib.load("rf_model.pkl")

st.set_page_config(
page_title="Game Commercial Success Predictor",
page_icon="🎮",
layout="centered"
)

# ---------- STATE ----------
if "show_popup" not in st.session_state:
    st.session_state.show_popup=False

if "pred_rate" not in st.session_state:
    st.session_state.pred_rate=None


# ---------- STYLE ----------
st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

[data-testid="stAppViewContainer"]{
padding-top:100px;
}

.stApp{
background:#2a475e;
}

.block-container{
max-width:760px;
padding:2rem;
background:#1b2838;
border-radius:20px;
}

html,body,p,span,label{
color:white !important;
}

h1{
text-align:center;
font-size:48px;
font-weight:800;
color:#66c0f4 !important;
}

[data-testid="stNumberInput"]{
background:#171a21;
padding:10px;
border-radius:14px;
border:1px solid #3b4d61;
}

input{
color:white !important;
-webkit-text-fill-color:white !important;
}

[data-testid="stButton"] button{
height:56px;
width:280px;
border:none;
border-radius:999px;
background:#66c0f4;
color:#171a21;
font-weight:800;
font-size:18px;
}


/* Overlay */
.overlay{
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
background:rgba(0,0,0,.70);
z-index:9998;
pointer-events:none;
}


/* Popup */
.popup{
position:fixed;
top:50%;
left:50%;
transform:translate(-50%,-50%);
z-index:9999;

width:1000px;
height:220px;

padding:55px;

border-radius:36px;
text-align:center;
box-shadow:0 0 55px rgba(0,0,0,.55);
}

.result{
background:linear-gradient(
135deg,
#0f172a,
#1e293b
);
border:3px solid #334155;
}

.popup-title{
font-size:62px;
font-weight:900;
margin-top:25px;
margin-bottom:35px;
}

.popup-text{
font-size:30px;
font-weight:700;
margin-bottom:0px;
}

.popup-rate{
font-size:72px;
font-weight:900;
margin:0;
}

[data-testid="stDialog"] .popup{
position:relative;
top:auto;
left:auto;
transform:none;
z-index:auto;
width:100%;
height:auto;
min-height:420px;
margin-bottom:14px;
}

[data-testid="stDialog"] .popup.result{
min-height:340px;
}

[data-testid="stDialog"] [role="dialog"]{
width:1000px;
max-width:95vw;
}

[data-testid="stDialog"] [data-testid="stButton"] button{
width:280px;
max-width:100%;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown(
"<h1>🎮 Game Commercial Success Predictor</h1>",
unsafe_allow_html=True
)

st.divider()


# ---------- INPUTS ----------
price_php=st.number_input("Price (PHP)",0.0,10000.0,1000.0)
discount=st.number_input("Discount %",0.0,100.0,10.0)
review=st.number_input("Review Score",0,100,80)
reviews=st.number_input("Total Reviews",0,1000000,5000)
peak=st.number_input("Peak Players",0,1000000,1000)
age=st.number_input("Game Age",0,50,5)
tags=st.number_input("Tag Count",1,20,5)


def compute_pred_rate(features: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        success_idx=-1
        if hasattr(model, "classes_"):
            classes=list(model.classes_)
            if 1 in classes:
                success_idx=classes.index(1)
            elif "1" in classes:
                success_idx=classes.index("1")

        base_vals=list(features.iloc[0].astype(float).values)
        mins=[0.0,0.0,0.0,0.0,0.0,0.0,1.0]
        maxs=[10000.0/56.0,100.0,100.0,1000000.0,1000000.0,50.0,20.0]
        steps=[0.5,0.5,0.5,500.0,500.0,0.25,0.25]

        samples=[base_vals]
        for idx,step in enumerate(steps):
            for direction in (-1.0,1.0):
                row=base_vals.copy()
                row[idx]=max(mins[idx],min(maxs[idx],row[idx]+direction*step))
                samples.append(row)

        sample_df=pd.DataFrame(samples,columns=[
            "Price_USD",
            "Discount_Pct",
            "Review_Score_Pct",
            "Total_Reviews",
            "24h_Peak_Players",
            "Game_Age",
            "Tag_Count"
        ])

        proba_vals=model.predict_proba(sample_df)[:,success_idx]
        p_raw=float(proba_vals.mean())
        pred_rate=p_raw*100
    elif hasattr(model, "decision_function"):
        decision=float(model.decision_function(features)[0])
        pred_rate=(1.0/(1.0+math.exp(-decision)))*100
    else:
        raw_pred=float(model.predict(features)[0])
        if 0.0<=raw_pred<=1.0:
            pred_rate=raw_pred*100
        elif 0.0<=raw_pred<=100.0:
            pred_rate=raw_pred
        else:
            pred_rate=100.0 if raw_pred>=0.5 else 0.0

    return max(0.0,min(100.0,pred_rate))


X=pd.DataFrame([[
    price_php/56,
    discount,
    review,
    reviews,
    peak,
    age,
    tags
]], columns=[
    "Price_USD",
    "Discount_Pct",
    "Review_Score_Pct",
    "Total_Reviews",
    "24h_Peak_Players",
    "Game_Age",
    "Tag_Count"
])


# ---------- Predict ----------
a,b,c=st.columns([1,1,1])

with b:
    if st.button("Predict Success"):
        st.session_state.pred_rate=compute_pred_rate(X)
        st.session_state.show_popup=True



# ---------- POPUP ----------
if st.session_state.show_popup:
    @st.dialog("Prediction Result")
    def show_popup_modal():
        st.session_state.pred_rate=compute_pred_rate(X)
        rate=st.session_state.pred_rate

        st.html(f"""
        <div class='popup result'>
            <div class='popup-title'>Estimated Success Rate</div>
            <div class='popup-rate'>{rate:.4f}%</div>
            <div class='popup-text'>Probability of strong commercial performance.</div>
        </div>
        """)
    show_popup_modal()