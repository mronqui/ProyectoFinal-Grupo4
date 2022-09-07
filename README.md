# Machine Learning project Grupo 4

1 - GoogleDrive: se debe tener permisos al repositorio del las imagenes (link)

2 - Notebook - Proyecto_Grupo_4_UTEC_Bootcamp_4Geeks : es donde analizamos datos, normalizamos imagenes y corrimos modelo el primer modelo, con dos dataset uno sin procesar y otro normalizado

3 - Notebook - Transfer_Learning_Model_2.ipynb : realizamos un segundo modelo con transfer learning, pero antes generamos más información usando data augmentation y generamos un nuevo modelo que mejoro mucho respecto al notebook anterior (salvamos el modelo como models/remito_model.h5)

4 - Notebook - Model_2_OCR.ipynb : usado para hacer las validaciones y generamos los funciones de OCR sobre las imagenes raw.

5 - /src/app.py   - Usamos el modelo generado en Transfer_Learning_Model_2.ipynb : models/remito_model.h5, con el cual clasificamos las imagenes y luego pasamos por las rutunas de OCR (utils.py)

6 - index.html - se genero la pagina HTML de prueba que se publicará para usar el modelo.

7 - Link a  Demo : http://ec2-3-92-234-136.compute-1.amazonaws.com:8080/
