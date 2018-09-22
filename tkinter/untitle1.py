import tkinter as tk
window=tk.Tk()
window.title('Natsu')
window.geometry('300x200')

var=tk.StringVar()

l=tk.Label(window,textvariable=var,bg='green',width=8,height=2)
l.pack()

on_hit=False

def hit_me():
    global on_hit
    if on_hit==False:
        on_hit=True
        var.set('you hit me')
    else:
        on_hit=False
        var.set('')


b=tk.Button(
    window,text='Button',width=5,height=1,
    command=hit_me
            )
b.pack()
window.mainloop()
