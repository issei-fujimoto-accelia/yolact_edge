# memo
chmod +x start_as_service.sh
chmod +x cec_start.sh
chmod +x cec_off.sh
chmod +x cec_status.sh


## user service
keyboardショートカットに以下の２つを設定
`systemctl --user stop myscript.service`
`systemctl --user start myscript.service`

`~/.config/systemd/user/myscript.service`にコピー
`cp myscript.service  ~/.config/systemd/user/myscript.service`
`systemctl --user daemon-reload`

### statusのチェック
systemctl --user status myscript.service

### reset
サービスが実行するscriptを修正した場合

systemctl --user reset-failed myscript.service

### reload
systemctl --user daemon-reload

### log
journalctl -u myscript.service -f



