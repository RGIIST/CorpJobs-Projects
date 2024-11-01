from pytube import YouTube


link = ''
YouTube(link).streams.filter(file_extension='mp4',res='1080p').first().download()
# YouTube('https://www.youtube.com/watch?v=Kj2eVZ4nINQ').streams.filter(progressive=True, file_extension='mp4',res='360p').first().download(PATH2SAVE)
# YouTube('https://www.youtube.com/watch?v=Hs0b10cejqg').streams.filter(only_audio=True).first().download() #only_video=True
