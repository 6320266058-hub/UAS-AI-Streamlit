# ==============================================================================
# APLIKASI MACHINE LEARNING & MONGODB (TES AKHIR SEMESTER)
# Fitur: Regresi, Klasifikasi, Clustering, MongoDB CRUD + Retrain, Komentar Publik
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from bson.objectid import ObjectId
import pymongo

# Scikit-learn: Datasets & Preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scikit-learn: Model Regresi, Klasifikasi, dan Clustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# Scikit-learn: Evaluasi
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard AI & ML - TAS",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Aplikasi Machine Learning End-to-End")
st.markdown("Implementasi **Regresi, Klasifikasi, Clustering, Retraining MongoDB, & Komentar Publik**")

# ------------------------------------------------------------------------------
# 2. KONEKSI MONGODB (CACHED)
# ------------------------------------------------------------------------------
@st.cache_resource
def init_mongo_connection():
    """Inisialisasi koneksi ke MongoDB Atlas menggunakan st.secrets."""
    try:
        uri = st.secrets["MONGODB_URI"]
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping') # Cek koneksi
        return client
    except Exception as e:
        st.error(f"❌ Gagal terhubung ke MongoDB: {e}")
        return None

client = init_mongo_connection()

# ------------------------------------------------------------------------------
# 3. FUNGSI LOAD DATASET (CACHED)
# ------------------------------------------------------------------------------
@st.cache_data
def load_sklearn_data(task_type, dataset_name):
    """Memuat toy datasets dari scikit-learn sesuai jenis task."""
    if task_type == "Regresi":
        if dataset_name == "Diabetes":
            data = datasets.load_diabetes()
        elif dataset_name == "California Housing":
            data = datasets.fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.feature_names, None

    elif task_type in ["Klasifikasi", "Clustering"]:
        if dataset_name == "Iris":
            data = datasets.load_iris()
        elif dataset_name == "Wine":
            data = datasets.load_wine()
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.feature_names, data.target_names

# ------------------------------------------------------------------------------
# 4. SIDEBAR PENGATURAN MODEL
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    
    # Pilih Jenis Task (Regresi, Klasifikasi, Clustering)
    task = st.radio(
        "🎯 Pilih Jenis Analisis:",
        ["Regresi", "Klasifikasi", "Clustering"]
    )
    
    # Pilih Dataset berdasarkan Task
    if task == "Regresi":
        dataset_choice = st.selectbox("📊 Pilih Dataset:", ["Diabetes", "California Housing"])
        model_choice = st.selectbox(
            "🧠 Pilih Model Regresi:",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest"]
        )
    elif task == "Klasifikasi":
        dataset_choice = st.selectbox("📊 Pilih Dataset:", ["Iris", "Wine", "Breast Cancer"])
        model_choice = st.selectbox(
            "🧠 Pilih Model Klasifikasi:",
            ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
        )
    else: # Clustering
        dataset_choice = st.selectbox("📊 Pilih Dataset:", ["Iris", "Wine", "Breast Cancer"])
        n_clusters = st.slider("🔢 Jumlah Cluster (K):", min_value=2, max_value=6, value=3)

    # Parameter Split Data (untuk Supervised Learning)
    if task != "Clustering":
        test_size = st.slider("📐 Test Size:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("🎲 Random State:", value=42, step=1)
    
    train_btn = st.button("🚀 Jalankan Model", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# 5. PROSES TRAINING & SESSION STATE
# ------------------------------------------------------------------------------
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

if train_btn:
    df, feature_names, target_names = load_sklearn_data(task, dataset_choice)
    X = df[feature_names]
    y = df['target']

    if task == "Regresi":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=random_state),
            "Lasso Regression": Lasso(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(random_state=random_state)
        }
        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.session_state.results = {
            "task": task, "dataset": dataset_choice, "model_name": model_choice,
            "y_test": y_test, "y_pred": y_pred,
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred)
        }

    elif task == "Klasifikasi":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "SVM": SVC(random_state=random_state)
        }
        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.session_state.results = {
            "task": task, "dataset": dataset_choice, "model_name": model_choice,
            "y_test": y_test, "y_pred": y_pred, "target_names": target_names,
            "acc": accuracy_score(y_test, y_pred),
            "prec": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "rec": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }

    elif task == "Clustering":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Reduksi dimensi ke 2D dengan PCA untuk visualisasi scatter plot
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        st.session_state.results = {
            "task": task, "dataset": dataset_choice, "model_name": f"K-Means (K={n_clusters})",
            "X_pca": X_pca, "clusters": clusters,
            "silhouette": silhouette_score(X_scaled, clusters),
            "inertia": kmeans.inertia_,
            "df_clustered": df.assign(Cluster=clusters)
        }

    st.session_state.model_trained = True

# ------------------------------------------------------------------------------
# 6. TABS UTAMA (4 TAB)
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Visualisasi Model",
    "📊 Evaluation Metrics",
    "🗄️ Database & Retrain (MongoDB)",
    "💬 Komentar Publik"
])

# === TAB 1: VISUALISASI ===
with tab1:
    if not st.session_state.model_trained:
        st.info("👈 Silakan pilih parameter di sidebar dan klik **Jalankan Model** terlebih dahulu.")
    else:
        res = st.session_state.results
        st.subheader(f"Hasil Visualisasi: {res['task']} - {res['dataset']} ({res['model_name']})")
        
        if res["task"] == "Regresi":
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(res["y_test"], res["y_pred"], alpha=0.7, edgecolors='k')
            ax.plot([res["y_test"].min(), res["y_test"].max()],
                    [res["y_test"].min(), res["y_test"].max()], 'r--', lw=2, label="Perfect Prediction")
            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            st.pyplot(fig)

        elif res["task"] == "Klasifikasi":
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                        xticklabels=res["target_names"], yticklabels=res["target_names"], ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        elif res["task"] == "Clustering":
            fig, ax = plt.subplots(figsize=(8, 4))
            scatter = ax.scatter(res["X_pca"][:, 0], res["X_pca"][:, 1], c=res["clusters"], cmap="viridis", alpha=0.8, edgecolors='k')
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.set_title(f"Visualisasi Cluster PCA 2D ({res['model_name']})")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            st.pyplot(fig)

# === TAB 2: EVALUATION METRICS ===
with tab2:
    if not st.session_state.model_trained:
        st.info("👈 Silakan jalankan model di sidebar untuk melihat metrik evaluasi.")
    else:
        res = st.session_state.results
        st.subheader(f"Metrik Evaluasi ({res['model_name']})")
        
        if res["task"] == "Regresi":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{res['mae']:.4f}")
            c2.metric("MSE", f"{res['mse']:.4f}")
            c3.metric("RMSE", f"{res['rmse']:.4f}")
            c4.metric("R² Score", f"{res['r2']:.4f}")

        elif res["task"] == "Klasifikasi":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{res['acc']:.4f}")
            c2.metric("Precision", f"{res['prec']:.4f}")
            c3.metric("Recall", f"{res['rec']:.4f}")
            c4.metric("F1-Score", f"{res['f1']:.4f}")
            
            st.markdown("#### 📋 Classification Report")
            st.dataframe(pd.DataFrame(res["report"]).transpose(), use_container_width=True)

        elif res["task"] == "Clustering":
            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score", f"{res['silhouette']:.4f}")
            c2.metric("Inertia (WCSS)", f"{res['inertia']:.2f}")
            
            st.markdown("#### 📋 Sampel Hasil Pengelompokan Data")
            st.dataframe(res["df_clustered"].head(15), use_container_width=True)

# === TAB 3: DATABASE & RETRAIN (MONGODB CRUD) ===
with tab3:
    st.header("🗄️ MongoDB CRUD & Retraining Model Iris")
    
    if client is None:
        st.warning("⚠️ Koneksi MongoDB belum aktif. Pastikan `.streamlit/secrets.toml` sudah dikonfigurasi.")
    else:
        db = client["test_db"]
        iris_col = db["iris_db"]

        # Fungsi ambil data dengan cache singkat
        @st.cache_data(ttl=5)
        def get_iris_docs():
            docs = list(iris_col.find())
            for d in docs:
                d["_id"] = str(d["_id"])
            return docs

        docs = get_iris_docs()
        st.metric("Total Dokumen di MongoDB (iris_db)", len(docs))

        col_form, col_data = st.columns([1, 1.5])
        
        # Form Tambah Data Iris
        with col_form:
            st.subheader("➕ Tambah Data Iris Baru")
            with st.form("form_iris"):
                s_len = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
                s_wid = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
                p_len = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
                p_wid = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)
                species = st.selectbox("Species", ["setosa", "versicolor", "virginica"])
                
                submit_iris = st.form_submit_button("💾 Simpan ke MongoDB", use_container_width=True)
                if submit_iris:
                    iris_col.insert_one({
                        "sepal_length": float(s_len),
                        "sepal_width": float(s_wid),
                        "petal_length": float(p_len),
                        "petal_width": float(p_wid),
                        "species": species,
                        "timestamp": datetime.now()
                    })
                    st.cache_data.clear()
                    st.success("✅ Data berhasil ditambahkan!")
                    st.rerun()

        # Tabel Data & Hapus Berdasarkan _id
        with col_data:
            st.subheader("📋 Data Tersimpan di MongoDB")
            if docs:
                df_mongo = pd.DataFrame(docs)
                st.dataframe(df_mongo, use_container_width=True, height=250)
                
                del_id = st.selectbox("Pilih `_id` dokumen untuk dihapus:", df_mongo["_id"].tolist())
                if st.button("🗑️ Hapus Dokumen Terpilih"):
                    iris_col.delete_one({"_id": ObjectId(del_id)})
                    st.cache_data.clear()
                    st.warning("🗑️ Dokumen berhasil dihapus!")
                    st.rerun()
            else:
                st.info("Belum ada data tambahan di collection `iris_db`.")

        st.divider()
        
        # Fitur Retraining Model (Sklearn 150 sampel + Data MongoDB)
        st.subheader("🔄 Retrain Model Klasifikasi Iris (Sklearn + MongoDB)")
        if st.button("⚡ Retrain Sekarang (Random Forest)", type="primary"):
            iris_sklearn = datasets.load_iris()
            X_base = pd.DataFrame(iris_sklearn.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
            y_base = pd.Series(iris_sklearn.target)

            if docs:
                df_new = pd.DataFrame(docs)
                mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
                X_new = df_new[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
                y_new = df_new["species"].map(mapping)
                
                X_combined = pd.concat([X_base, X_new], ignore_index=True)
                y_combined = pd.concat([y_base, y_new], ignore_index=True)
            else:
                X_combined, y_combined = X_base, y_base

            X_tr, X_te, y_tr, y_te = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
            rf_retrain = RandomForestClassifier(random_state=42)
            rf_retrain.fit(X_tr, y_tr)
            preds = rf_retrain.predict(X_te)

            st.success(f"✅ Retraining selesai menggunakan total **{len(X_combined)}** data (150 Sklearn + {len(docs)} MongoDB)!")
            
            c_acc, c_fi = st.columns([1, 1.5])
            with c_acc:
                st.metric("Akurasi Model Terbaru", f"{accuracy_score(y_te, preds)*100:.2f}%")
                st.text("Classification Report:")
                st.dataframe(pd.DataFrame(classification_report(y_te, preds, output_dict=True)).transpose())
            with c_fi:
                fig_fi, ax_fi = plt.subplots(figsize=(6, 3.5))
                sns.barplot(x=rf_retrain.feature_importances_, y=X_combined.columns, ax=ax_fi, palette="viridis")
                ax_fi.set_title("Feature Importance Setelah Retrain")
                st.pyplot(fig_fi)

# === TAB 4: KOMENTAR PUBLIK (MONGODB) ===
with tab4:
    st.header("💬 Kolom Komentar Publik")
    st.markdown("Silakan tinggalkan masukan, ulasan, atau catatan pengujian aplikasi di bawah ini.")
    
    if client is None:
        st.warning("⚠️ Koneksi MongoDB belum aktif.")
    else:
        db = client["test_db"]
        comment_col = db["public_comments"]

        # Form Input Komentar
        with st.form("form_komentar", clear_on_submit=True):
            nama_user = st.text_input("Nama Anda", placeholder="Contoh: Budi Santoso")
            pesan_user = st.text_area("Komentar", placeholder="Tulis komentar Anda di sini...")
            kirim_btn = st.form_submit_button("📨 Kirim Komentar")
            
            if kirim_btn:
                if nama_user.strip() and pesan_user.strip():
                    comment_col.insert_one({
                        "nama": nama_user.strip(),
                        "komentar": pesan_user.strip(),
                        "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    st.success("✅ Terima kasih! Komentar Anda telah tersimpan.")
                    st.rerun()
                else:
                    st.error("⚠️ Nama dan komentar tidak boleh kosong.")

        st.divider()
        st.subheader("Daftar Komentar Pengunjung")
        
        # Tampilkan Komentar Terbaru di Atas
        daftar_komentar = list(comment_col.find().sort("_id", -1))
        if daftar_komentar:
            for item in daftar_komentar:
                with st.chat_message("user"):
                    st.markdown(f"**{item.get('nama', 'Anonim')}** • *{item.get('waktu', '')}*")
                    st.write(item.get('komentar', ''))
        else:
            st.info("Belum ada komentar. Jadilah yang pertama berkomentar!")
MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority"