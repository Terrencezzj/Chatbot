Can be used to get information of the price, turnover, capitalization of stocks.

Dependencies:
	iexfinance
	rasa_nlu
	spacy
	wxpy

Usage:
	Just run the main.py
	You need to login in the wechat by QRcode.
	The conversation initially holds in file_helper, but you can easily change it by modified @bot.register(bot.file_helper) to friend or other chat windows. 