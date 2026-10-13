# บทพูดนำเสนอโครงงาน Machine Learning
## การทำนายการลาออกของพนักงาน (Employee Attrition Prediction)

---

## 1. คำสั่งที่ใช้ในงานโปรเจกต์ พร้อมคำอธิบาย

### 1.1 การโหลดและเตรียมข้อมูล (Data Loading & Preprocessing)
```python
# 1. โหลดข้อมูล
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# 2. ตัดคอลัมน์ที่ไม่จำเป็น (ค่าคงที่และรหัสพนักงาน)
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df_cleaned = df.drop(columns=drop_cols)

# 3. แปลงคลาสเป้าหมาย Attrition เป็นตัวเลข (Yes = 1, No = 0)
df_cleaned['Attrition'] = df_cleaned['Attrition'].map({'Yes': 1, 'No': 0})

# 4. แปลงข้อมูลเชิงกลุ่มด้วย One-Hot Encoding
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
```
* **คำอธิบาย:**
  * ใช้ `pd.read_csv` อ่านข้อมูลเข้าสู่โปรแกรม
  * ใช้ `.drop()` ตัดคอลัมน์ที่มีค่าเหมือนกันทุกคนออก เพราะไม่ส่งผลต่อการเรียนรู้ของโมเดล
  * ใช้ `.map()` และ `pd.get_dummies()` แปลงตัวอักษร เช่น แผนกและตำแหน่งงาน ให้เป็นตัวเลข 0 และ 1

---

### 1.2 การแบ่งข้อมูลและปรับสเกล (Train-Test Split & Scaling)
```python
# แยก Feature (X) และ Target (y)
X = df_encoded.drop(columns=['Attrition'])
y = df_encoded['Attrition']

# แบ่งข้อมูล 80% สำหรับสอน (Train) และ 20% สำหรับทดสอบ (Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ปรับสเกลข้อมูลให้อยู่ในมาตรฐานเดียวกัน
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
* **คำอธิบาย:**
  * ใช้ `train_test_split` โดยกำหนด `stratify=y` เพื่อคงสัดส่วนคนลาออก (16%) ให้เท่ากันทั้งในชุด Train และ Test
  * ใช้ `StandardScaler` ปรับข้อมูลตัวเลขที่มีสเกลต่างกัน (เช่น อายุ 20–60 และเงินเดือน 2,000–20,000) ให้อยู่ในช่วงมาตรฐาน

---

### 1.3 การฝึกสอนโมเดล (Model Training)
```python
# 1. Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# 2. Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

# 3. Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# 4. Support Vector Machine (SVM)
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
```
* **คำอธิบาย:**
  * ใช้คำสั่ง `.fit()` ฝึกสอน 3 โมเดลหลักตามโจทย์ ได้แก่ Logistic Regression (โมเดลเชิงเส้น), Random Forest (โมเดลกลุ่มต้นไม้ตัดสินใจ), Gaussian Naive Bayes (โมเดลความน่าจะเป็น) และเสริมด้วย SVM

---

### 1.4 การวัดประสิทธิภาพ (Evaluation Metrics)
```python
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
```
* **คำอธิบาย:**
  * คำนวณค่าตัวชี้วัด Accuracy, Precision, Recall, F1-Score และแสดงผล Confusion Matrix

---

## 2. ตัวอย่างรายการข้อมูลใน Dataset ที่ใช้

* **ภาพรวม:** ชุดข้อมูลมี 1,470 แถว 35 คุณลักษณะ ข้อมูลไม่มี Missing Value
* **ตัวแปรเป้าหมาย (Target):** `Attrition` (`Yes` = ลาออก 237 คน / `No` = ไม่ลาออก 1,233 คน)

### ตัวอย่างข้อมูลจริง 3 แถวจากชุดข้อมูล:

| อายุ (Age) | แผนก (Department) | ตำแหน่ง (JobRole) | เงินเดือน (MonthlyIncome) | ล่วงเวลา (OverTime) | อายุงาน (TotalWorkingYears) | สถานะลาออก (Attrition) |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **41** | Sales | Sales Executive | $5,993 | **Yes** | 8 ปี | **Yes (ลาออก)** |
| **49** | Research & Development | Research Scientist | $5,130 | **No** | 10 ปี | **No (ไม่ลาออก)** |
| **37** | Research & Development | Laboratory Technician | $2,090 | **Yes** | 7 ปี | **Yes (ลาออก)** |

---

## 3. ผลลัพธ์ที่ได้จากการรันคำสั่ง

### 3.1 ตารางเปรียบเทียบประสิทธิภาพโมเดล (บน Test Data 294 ตัวอย่าง)

| โมเดล (Model) | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **1. Logistic Regression** | **86.05%** | 61.54% | 34.04% | **43.84%** *(สูงสุด)* |
| **2. Random Forest** | 83.33% | 41.67% | 10.64% | 16.95% |
| **3. Naive Bayes (Gaussian)** | 61.56% | 25.00% | **70.21%** *(สูงสุด)* | 36.87% |
| *(เสริม) **SVM (RBF Kernel)*** | 86.73% | **90.00%** *(สูงสุด)* | 19.15% | 31.58% |

---

### 3.2 ผลลัพธ์ Confusion Matrix (แจกแจงจำนวนจริง)
*(ใน Test Set: คนไม่ลาออกจริง 247 คน, คนลาออกจริง 47 คน)*

* **Logistic Regression:**
  * ทายคนไม่ลาออกถูก: $237$ คน
  * ทายคนลาออกถูก (TP): $16$ คน
  * ทายหลุดว่าไม่ลาออก (FN): $31$ คน
  * **สรุป:** ได้ค่า **F1-Score สูงสุดที่ 43.84%** มีความสมดุลระหว่าง Precision และ Recall มากที่สุด

* **Random Forest:**
  * ทายคนไม่ลาออกถูก: $240$ คน
  * ทายคนลาออกถูก (TP): $5$ คน
  * **สรุป:** มี Accuracy สูง (83.33%) แต่ Recall ค่อนข้างต่ำเนื่องจากปัญหา Class Imbalance

* **Naive Bayes (Gaussian):**
  * ทายคนไม่ลาออกถูก: $148$ คน
  * ทายคนลาออกถูก (TP): **$33$ คน จากทั้งหมด $47$ คน** (หลุดไปเพียง 14 คน)
  * **สรุป:** ได้ค่า **Recall สูงสุดถึง 70.21%** เหมาะสำหรับใช้เป็นระบบเตือนภัยล่วงหน้าเพื่อดักจับคนที่จะลาออกให้ได้มากที่สุด
