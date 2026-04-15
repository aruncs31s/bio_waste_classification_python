sudo cp ./waste_classifier.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable waste_classifier.service
sudo systemctl start waste_classifier.service
