import fasttext
import fasttext.util
fasttext.util.download_model('es', if_exists='ignore')  # Espanish
ft = fasttext.load_model('cc.es.100.bin')

print(ft.get_dimension())
fasttext.util.reduce_model(ft, 100)
print(ft.get_dimension())

ft.save_model('cc.es.100.bin')