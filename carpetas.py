import os

path = os.getcwd()

for x in range(1, 21):
    prefijo = (10 * x) - 9
    sufijo = 10 * x
    print(f"{prefijo} - {sufijo}")
    carpeta = os.path.join(path, f"{prefijo}-{sufijo}")
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
