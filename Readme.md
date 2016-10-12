# Readme emai

Dies ist die Realisierung der Thesis "Analyse und Bestimmung von Emotionen in textbasierten Konversationen mittels Active Learning" von Martin Juhasz.

## Installation

Entpacken Sie die Datei Sources.zip. Darin enthalten ist der Ordner "emai" (Codename des Projektes), welcher die Projektdateien, Konfigurationsdateien und Quelldateien des Clients (Ordner 'emai/client') und des Servers (Ordner 'emai/emai') enthält.


### Server
Für das Ausführen des Servers wird Python mit Mindestversion 3.4 benötigt. Ebenso wird ein laufender Mongodb-Server mit der Mindestversion 3.2 benötigt.
Zudem sollte zur Aufzeichnung von Aufnahmen ffmpeg kompiliert vorliegen und der Pfad später in den Einstellungen konfiguriert werden. 
Sind diese Vorraussetzungen erfüllt können die externen Abhängigkeiten mit pip installiert werden. Dazu gehen Sie zuerst in den Hauptordner `emai` und führen Sie folgenden Befehl aus.
```
pip install -r requirements.txt
```

Anschließend muss auch noch der Server selbst als Paket mit Distutils registriert werden. Dies erstellt einen Symlink anstatt das gesamte Paket zu kopieren
```
python setup.py develop
``` 

### Client
Der Client muss nicht zwingend installiert und kompiliert werden, um ihn zu starten. Eine bereits kompilierte Version des statischen Clients liegt mit bei und kann direkt gestartet werden (Siehe Ausführen). Möchte man den Client kompilieren sollte zunächst Node.js und damit auch NPM installiert werden.
Dazu gehen Sie zuerst in den Hauptordner `emai` und führen Sie folgenden Befehl aus.
```
npm install
```

Anschließend kann eine eine lauffähige Version des Clients in den Ordner `static` erzeugt werden.
```
npm run compile
```

## Ausführen

### Server

Bevor Sie den Server ausführen müssen zuerst in der `config.ini` Einstellungen bezüglich der Datenbankverbindung (`persistence`), des Pfads von FFMpeg (`recording`), und Logindaten von Twitch.tv(`twitch`) vornehmen. Sollten Sie noch keinen Twitch Login haben können Sie diesen auf der Webseite von Twtich erstellen und dort das O-Auth Token erhalten.

Danach kann der Server gestartet werden. 
```
python emai/server.py
```

### Client
Sollten Sie den Client nicht selbst kompiliert haben, kann die bereits kompilierte Version im Ordner `emai/static` verwendet werden. Diese stellt eine Verbindung zum Server mit der Adresse `0.0.0.0:8082` her. Sollten Sie diese Einstellungen im Server geändert haben, müssen sie den Client selbst mit der passenden IP und Port kompilieren. Die Einstellung hierfür findet sich im Ordner `emai/client/api/emai.js`.

Um den Client zu starten müssen die Dateien von einem einfachen WebServer geladen werden. Hierzu kann beispielsweise mit NPM ein solcher installiert werden.
```
npm install http-server -g
```

Und dann der Client im Ordner `emai/static` bereit gestellt werden.
```
cd static
http-server -p 8081
```

Alternativ kann auch der integrierte Python Server verwendet werden. Dieser unterstützt allerdings nicht die Übertragung von Videodateien und somit können die Aufnahmen nicht abgespielt werden.
```
python -m http.server 8081
```

## Beispieldaten
Es können mitgelieferte Beispieldaten geladen werden, um keine eigenen Aufnahmen oder Klassifikationen vornehmen zu müssen. Diese Beispieldaten waren auch Teil der Evaluation. Entpacken Sie dazu den Ordner `Samples.zip`. Darin enthalten sind zum einen die Videodateien und die Datenbankeinträge.
Kopieren sie den Ordner `videos` in den statischen Ordner des Clients ihres Client-Servers nach `static/videos`. Die mitgelieferten Datenbankeinträge können mit folgendem Befehl geladen werden. Dabei ist `<database name>` der Name der gewünschten Zieldatenbank (Default: `emai`) und <database folder> der Ordner `database` den Sie gerade entpackt haben.
```
mongorestore --drop -d <database name> <database folder>
```
Danach können Server und Client gestartet werden. 


