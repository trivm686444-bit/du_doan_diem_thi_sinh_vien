import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dự Đoán Điểm Thi AI", page_icon="🎓", layout="centered")

st.title("🎓 HỆ THỐNG TRÍ TUỆ NHÂN TẠO DỰ ĐOÁN ĐIỂM THI")
st.markdown("Ứng dụng web tương tác trực tiếp giúp dự đoán điểm thi cuối kỳ của sinh viên.")

@st.cache_resource
def load_and_train():
    df = pd.read_excel('Exam_Score_Prediction_tieng_viet_P7.xlsx')
    X = df.drop(columns=['Mã học sinh', 'Điểm thi'])
    y = df['Điểm thi']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    model = Pipeline(steps=[
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    return model

model = load_and_train()

st.sidebar.header("📋 Nhập Thông Tin Sinh Viên")
so_gio_hoc = st.sidebar.slider("Số giờ học trên lớp/ngày", 1.0, 12.0, 6.0)
gioi_tinh = st.sidebar.selectbox("Giới tính", ["Nam", "Nữ"])
khoa_hoc = st.sidebar.selectbox("Khóa học", ["Cử nhân Ứng dụng Máy tính (BCA)", "Kỹ thuật Điện", "Khoa học Máy tính", "Quản trị Kinh doanh"])
gpa = st.sidebar.slider("Điểm GPA các kỳ trước", 0.0, 10.0, 8.0)
thoi_gian_tu_hoc = st.sidebar.slider("Thời gian tự học ngoài giờ (giờ/tuần)", 0.0, 40.0, 15.0)
muc_cang_thang = st.sidebar.selectbox("Mức độ căng thẳng", ["Thấp", "Trung bình", "Cao"])

input_data = pd.DataFrame([{
    'Tuổi': 20,
    'Giới tính': i_tinh,
    'Khóa học': khoa_hoc,
    'Số giờ học': so_gio_hoc,
    'Tỷ lệ đi học (%)': 90.0,
    'Có internet': 'Có',
    'Số giờ ngủ': 7.5,
    'Chất lượng giấc ngủ': 'Tốt',
    'Phương pháp học': 'Học thêm / Luyện thi',
    'Đánh giá cơ sở vật chất': 'Cao',
    'Độ khó kỳ thi': 'Trung bình',
    'Thời gian tự học ngoài giờ (giờ/tần)': thoi_gian_tu_hoc,
    'Mức độ căng thẳng': muc_cang_thang,
    'Điểm trung bình các kỳ trước (GPA)': gpa
}])

if st.sidebar.button("🚀 Dự Đoán Điểm Thi Ngay"):
    diem_du_doan = model.predict(input_data)[0]
    st.success("✨ Phân tích dữ liệu thành công!")
    st.metric(label="🎯 Điểm Thi Dự Kiến (Thang điểm 100)", value=f"{diem_du_doan:.2f} điểm")
    if diem_du_doan >= 80:
        st.balloons()
        st.info("💡 **Nhận xét từ AI:** Học lực xuất sắc! Khả năng đạt điểm cao rất lớn.")
    else:
        st.warning("⚠️ **Nhận xét từ AI:** Cần tăng thời gian tự học ngoài giờ để cải thiện kết quả.")
