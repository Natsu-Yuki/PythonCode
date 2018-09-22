import tkinter as tk
window=tk.Tk()
window.title('Natsu')
window.geometry('300x200')



var=tk.StringVar()

l=tk.Label(window,bg='yellow',width=20)

l.pack()

def se(v):
    l.config(text='you have select '+v)

s=tk.Scale(
    window,label='try me',from_=5,
           to=11,orient=tk.HORIZONTAL,length=200,
           showvalue=1,tickinterval=3,resolution=0.01,
            command=se
           )
s.pack()


window.mainloop()
