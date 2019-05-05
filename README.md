# Tensorflow-Projekte

Dieses Projekt soll einige praktische Beispiele von Tensorflow enthalten.

## (CNN) Bilder laden und CNN trainieren

Mittels diesen Code koennen Bilder aus einem Ordner gelesen werden und in das 
Tensorflow Datei Format formatiert werden. 
Dieser Code wurde fuer Python 2.7 geschrieben und getestet. Fuer andere Versionen muessten dementsprechende
Anpassungen durchgefuehrt werden

Dem Nutzer bleibt die Option Testdaten zufaellig aus den Trainingsbildern zu waehlen,
die Testbilder explizit anzugeben oder einen eigenen Pfad zu den Testbilder zu geben.
Dabei ist es wichtig, dass der Ordner des Trainingsbilder folgende Struktur aufweist:
./Trainingsbilder/
- /ersteKlasse/
- - Bild1.png
- - Bild2.png
- - ...
- /zweiteKlasse/
- - Bild1.png
- - Bild2.png
- - ...
- ...

WARNUNG: Ordner fuer Trainings- und Testbilder muessen soviele Unterordner wie Klassen haben.

Anschlie{\ss}end wird ein CNN definiert und anhand der Trainingsbilder trainiert. Das CNN
hat insgesamt drei Conolutional Layer und drei Pooling Layer

Dieser Code ist stark angelehnt von folgendem Projekt:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

### ToDo

- [ ] Pipeline modifizieren
- [ ] Skript zum manuellen Testen der .pb
- [ ] Möglichkeit direkt eine .pb zu erzeugen ohne .pbtxt

