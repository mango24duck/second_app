# app.py

import streamlit as st
import pandas as pd
from sklearn.svm import SVC # Changed from BernoulliNB to SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# -------------------- 페이지 설정 --------------------
st.set_page_config(
    page_title="수능 영어 정답 예측기",
    page_icon="✏️"
)

# 1. 데이터 로드 및 전처리
# 파일명을 eng_final.csv로 지정합니다.
file_path = "eng_final.csv"

@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"'{file_path}' 파일을 찾을 수 없습니다. 파일이 앱과 같은 폴더에 있는지 확인해주세요.")
        st.stop()
        
    # 'year', 'month', 'number' 컬럼의 문자열 제거 및 정수형 변환
    df['year'] = df['year'].str.replace('y', '').astype(int)
    df['month'] = df['month'].str.replace('m', '').astype(int)
    df['number'] = df['number'].str.replace('no', '').astype(int)
    
    return df

# 2. 모델 학습 함수 (두 가지 케이스 모두 학습)
@st.cache_resource
def train_all_models(df):
    # 'month'와 'number'를 모두 사용하는 경우
    X_full_encoded = pd.get_dummies(df[['month', 'number']].astype('category'), columns=['month', 'number'], prefix=['month', 'number'])
    y = df['answer']
    # Changed model to SVC with probability=True
    full_model = SVC(probability=True, random_state=42)
    full_model.fit(X_full_encoded, y)
    
    # 'number'만 사용하는 경우
    X_num_encoded = pd.get_dummies(df[['number']].astype('category'), columns=['number'], prefix=['number'])
    y_num = df['answer']
    # Changed model to SVC with probability=True
    comp_model = SVC(probability=True, random_state=42)
    comp_model.fit(X_num_encoded, y_num)
    
    return {
        'full': {
            'model': full_model,
            'features': X_full_encoded.columns
        },
        'comp': {
            'model': comp_model,
            'features': X_num_encoded.columns
        }
    }

# 3. 예측 결과를 DataFrame으로 생성하는 함수
def get_predictions_df(model_data, selected_month_str):
    model = model_data['model']
    feature_columns = model_data['features']
    
    # 18번부터 45번까지 문제 번호
    question_numbers = range(18, 46)
    results = []
    classes = model.classes_
    
    # '종합' 선택 시, 예측에 사용할 피처는 'number'만 사용
    if selected_month_str == '종합':
        for number in question_numbers:
            predict_data = pd.DataFrame([{'number': number}])
            X_pred_row = pd.get_dummies(predict_data, columns=['number'], prefix=['number'])
            X_pred_row = X_pred_row.reindex(columns=feature_columns, fill_value=0)
            probabilities = model.predict_proba(X_pred_row)[0]
            
            row_dict = {"문제 번호": number}
            for i, prob in enumerate(probabilities):
                row_dict[f"{classes[i]}번 답 확률"] = f"{prob * 100:.2f}%"
            results.append(row_dict)
    else: # 특정 월 선택 시, 예측에 사용할 피처는 'month'와 'number' 모두 사용
        for number in question_numbers:
            # 선택된 월 문자열을 정수형으로 변환 (예: '3월' -> 3, '수능' -> 11)
            month_to_use = 11 if selected_month_str == '수능' else int(selected_month_str.replace('월', ''))
            predict_data = pd.DataFrame([{'month': month_to_use, 'number': number}])
            X_pred_row = pd.get_dummies(predict_data, columns=['month', 'number'], prefix=['month', 'number'])
            X_pred_row = X_pred_row.reindex(columns=feature_columns, fill_value=0)
            probabilities = model.predict_proba(X_pred_row)[0]
            
            row_dict = {"문제 번호": number}
            for i, prob in enumerate(probabilities):
                row_dict[f"{classes[i]}번 답 확률"] = f"{prob * 100:.2f}%"
            results.append(row_dict)
    
    results_df = pd.DataFrame(results)
    return results_df

# Streamlit 앱 시작
st.title("수능 영어 정답 예측 모델 📝")
st.write("최근 7년간 모의고사와 수능 정답 데이터를 SVM 분류 방식으로 학습시킨 머신 러닝 모델입니다.") # Updated description
st.markdown("---")

# 데이터 로드
data = load_and_preprocess_data(file_path)

# 모든 모델 학습 (캐싱으로 효율성 증대)
models = train_all_models(data)

# 3. 모델 학습 방식 표시
st.header("모델 학습 결과")
st.info("전체 데이터를 사용하여 모델을 학습했습니다.")
st.markdown("_(별도의 평가 데이터셋 없이 모든 데이터를 학습에 사용했습니다)_")
st.markdown("---")

# 4. 사용자 입력 받기
st.header("월별 정답 확률 예측")

# 사용자가 선택할 수 있는 월 리스트를 '월'을 포함한 형태로 변경
numeric_months = sorted([m for m in [3, 4, 6, 7, 9, 10, 11] if m != 11])
formatted_months = [f'{m}월' for m in numeric_months]
available_months = ['종합'] + formatted_months + ['수능']

selected_month_str = st.selectbox(
    "모의고사 시행 월을 선택하세요.",
    options=available_months
)

# 5. 선택된 월에 대한 예측 결과 표시
if selected_month_str:
    if selected_month_str == '종합':
        st.subheader(f"전체 모의고사 데이터 기반 정답 확률 예측")
        model_to_use = models['comp']
    else:
        st.subheader(f"{selected_month_str} 모의고사 정답 확률 예측" if selected_month_str != '수능' else f"수능 정답 확률 예측")
        model_to_use = models['full']
    
    results_df = get_predictions_df(model_to_use, selected_month_str)
    
    # 데이터프레임의 확률 컬럼만 선택
    prob_columns = [col for col in results_df.columns if '확률' in col]
    
    # 하이라이트 스타일 함수
    def highlight_max(s):
        s_numeric = s.str.replace('%', '').astype(float)
        is_max = s_numeric == s_numeric.max()
        return ['background-color: #d1e7ff' if v else '' for v in is_max]
        
    # 하이라이트 스타일 적용
    styled_df = results_df.style.apply(highlight_max, subset=prob_columns, axis=1)
    
    st.dataframe(styled_df, hide_index=True)
