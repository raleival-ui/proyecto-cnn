import streamlit as st
import tensorflow as tf
import os
import tempfile
from pathlib import Path
from app.config import Config

class ModelManager:
    def __init__(self):
        self.config = Config()
        self.max_file_size = 3 * 1024 * 1024 * 1024  # 3GB
        
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = None
        if 'model_info' not in st.session_state:
            st.session_state.model_info = None
        if 'model_path' not in st.session_state:
            st.session_state.model_path = None
    
    def load_model_interface(self):
        """Interface de Streamlit para cargar modelos con límite de 3GB"""
        st.subheader("Cargar Modelo (Hasta 3GB)")
        
        current_model = self.get_current_model()
        if current_model is not None:
            st.success("Modelo ya cargado y listo para usar")
            if st.session_state.model_info:
                info = st.session_state.model_info
                st.info(f"Modelo activo: {info.get('original_name', 'modelo')} ({info.get('size_mb', 0):.1f} MB)")
        
        # Mostrar límite actualizado
        st.info("✅ Límite de archivo configurado: 3GB (3072MB)")
        
        st.markdown("**Opción 1: Subir Archivo (Hasta 3GB)**")
        
        with st.expander("Instrucciones para archivos grandes (2-3GB)"):
            st.markdown("""
            **Para modelos muy grandes (2-3GB):**
            1. La aplicación está configurada para aceptar hasta 3GB
            2. El proceso puede tomar 5-10 minutos
            3. No cierres la pestaña durante la carga
            4. Se mostrará progreso en tiempo real
            5. Se usa almacenamiento temporal eficiente
            
            **Requisitos:**
            - Conexión estable a internet
            - Suficiente RAM en el servidor
            - Paciencia durante la carga
            """)
        
        uploaded_file = st.file_uploader(
            "Sube tu modelo (.h5, .keras)", 
            type=["h5", "keras"],
            help="Archivos hasta 3GB soportados - Configuración especial activada",
            key="model_uploader"
        )
        
        if uploaded_file:
            # Verificar tamaño
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            if uploaded_file.size > self.max_file_size:
                st.error(f"❌ Archivo demasiado grande: {file_size_mb:.1f} MB (Límite: 3072 MB)")
                return current_model
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Archivo", uploaded_file.name)
            with col2:
                st.metric("Tamaño", f"{file_size_mb:.1f} MB")
            with col3:
                if file_size_mb > 2000:
                    st.metric("Estado", "⚠️ Muy Grande")
                elif file_size_mb > 1000:
                    st.metric("Estado", "🔶 Grande")
                else:
                    st.metric("Estado", "✅ Normal")
            
            # Advertencia para archivos muy grandes
            if file_size_mb > 2000:
                st.warning("""
                **Archivo muy grande detectado (2-3GB):**
                - Tiempo de carga estimado: 5-10 minutos
                - Usa una conexión estable
                - No cierres esta pestaña
                """)
            
            if st.button("🚀 Cargar Modelo Grande", type="primary", key="btn_upload_large"):
                with st.spinner(f"Cargando modelo grande ({file_size_mb:.1f} MB)..."):
                    loaded_model = self.load_model_from_upload_optimized(uploaded_file)
                    if loaded_model:
                        st.balloons()
                        st.rerun()
        
        # Resto de tu código para otras opciones...
        st.markdown("**Opción 2: Ruta del Modelo (Recomendado para archivos grandes)**")
        
        model_path = st.text_input(
            "Ruta completa al modelo", 
            value=str(self.config.DEFAULT_MODEL_PATH),
            help="Ejemplo: C:/ruta/completa/modelo.h5 o /home/user/modelo.h5",
            key="model_path_input"
        )
        
        path_exists = os.path.exists(model_path) if model_path else False
        if model_path and path_exists:
            try:
                file_size = os.path.getsize(model_path)
                file_size_mb = file_size / (1024 * 1024)
                
                if file_size_mb > 2000:
                    st.warning(f"Archivo muy grande encontrado ({file_size_mb:.1f} MB)")
                else:
                    st.success(f"Archivo encontrado ({file_size_mb:.1f} MB)")
                    
            except:
                st.warning("No se puede obtener información del archivo")
        elif model_path:
            st.error("Archivo no encontrado en la ruta especificada")
        
        if st.button("Cargar desde Ruta", disabled=not path_exists, key="btn_path"):
            if path_exists:
                with st.spinner("Cargando modelo desde ruta..."):
                    loaded_model = self.load_model_from_path(model_path)
                    if loaded_model:
                        st.experimental_rerun()
            else:
                st.error("Especifica una ruta válida")
        
        return current_model
    
    def load_model_from_upload_optimized(self, uploaded_file):
        """Versión optimizada para archivos grandes"""
        if uploaded_file is None:
            return None
            
        try:
            file_size = uploaded_file.size
            st.info(f"Procesando archivo grande: {uploaded_file.name} ({file_size / (1024*1024*1024):.2f} GB)")
            
            # Progreso más detallado para archivos grandes
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                tmp_file_path = tmp_file.name
                
                # Escribir en chunks optimizados
                chunk_size = 10 * 1024 * 1024  # 10MB chunks para archivos grandes
                bytes_written = 0
                
                uploaded_file.seek(0)
                
                while True:
                    chunk = uploaded_file.read(chunk_size)
                    if not chunk:
                        break
                    
                    tmp_file.write(chunk)
                    bytes_written += len(chunk)
                    
                    # Actualizar progreso
                    progress = min(bytes_written / file_size, 1.0)
                    progress_bar.progress(progress)
                    
                    # Mostrar información detallada
                    mb_written = bytes_written / (1024 * 1024)
                    mb_total = file_size / (1024 * 1024)
                    status_text.text(f"Progreso: {mb_written:.0f} / {mb_total:.0f} MB ({progress:.1%})")
            
            progress_bar.empty()
            status_text.empty()
            
            # Cargar modelo
            st.info("🔄 Cargando modelo en memoria...")
            model = self._load_model_from_path(tmp_file_path)
            
            if model:
                # Guardar permanentemente
                permanent_path = self.config.MODELS_DIR / f"uploaded_{uploaded_file.name}"
                permanent_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Guardar con barra de progreso
                with st.spinner("Guardando modelo permanentemente..."):
                    model.save(str(permanent_path))
                
                # Actualizar session state
                st.session_state.model_loaded = model
                st.session_state.model_path = str(permanent_path)
                st.session_state.model_info = {
                    'source': 'upload',
                    'original_name': uploaded_file.name,
                    'path': str(permanent_path),
                    'size_mb': file_size / (1024 * 1024),
                    'size_gb': file_size / (1024 * 1024 * 1024),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'total_params': model.count_params()
                }
                
                st.success(f"✅ Modelo grande cargado exitosamente: {file_size/(1024*1024*1024):.2f} GB")
            
            # Limpiar temporal
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            
            return model
            
        except Exception as e:
            st.error(f"❌ Error procesando archivo grande: {str(e)}")
            # Limpiar en caso de error
            try:
                os.unlink(tmp_file_path)
            except:
                pass
            return None

    # Mantén tus otros métodos existentes...
    @st.cache_resource
    def _load_model_from_path(_self, model_path):
        """Cargar modelo desde una ruta específica (versión interna con cache)"""
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error cargando modelo: {str(e)}")
            return None

    def load_model_from_path(self, model_path):
        """Cargar modelo desde ruta y guardarlo en session_state"""
        try:
            file_size = os.path.getsize(model_path)
            file_size_gb = file_size / (1024*1024*1024)
            
            if file_size_gb > 2:
                st.info(f"🔄 Cargando modelo grande ({file_size_gb:.2f} GB)...")
            else:
                st.info(f"Cargando modelo ({file_size / (1024*1024):.1f} MB)...")
            
            model = self._load_model_from_path(model_path)
            
            if model:
                st.session_state.model_loaded = model
                st.session_state.model_path = model_path
                st.session_state.model_info = {
                    'source': 'path',
                    'path': model_path,
                    'size_mb': file_size / (1024*1024),
                    'size_gb': file_size_gb,
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape,
                    'total_params': model.count_params()
                }
                
                if file_size_gb > 2:
                    st.success(f"✅ Modelo grande cargado: {file_size_gb:.2f} GB")
                else:
                    st.success("Modelo cargado exitosamente")
            
            return model
        except Exception as e:
            st.error(f"Error cargando modelo: {str(e)}")
            return None

    def get_current_model(self):
        return st.session_state.model_loaded

    def clear_model(self):
        st.session_state.model_loaded = None
        st.session_state.model_info = None
        st.session_state.model_path = None
        st.cache_resource.clear()
