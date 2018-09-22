import tkinter as tk
window=tk.Tk()
window.title('Natsu')
window.geometry('300x200')

l=tk.Label(window,bg='yellow',width=20)
l.pack()

def se():
    if((var1.get()==1)&(var2.get()==1)):
        l.config(text='python  & c')
    elif((var1.get()==1)&(var2.get()==0)):
        l.config(text='python  ')
    elif ((var1.get() == 0) & (var2.get() == 1)):
        l.config(text='c  ')
    else:
        l.config(text='fuck  ')



var1=tk.IntVar()
c1=tk.Checkbutton(window,text='python',
                  variable=var1,onvalue=1,
                  offvalue=0,command=se
                  )
var2=tk.IntVar()
c2=tk.Checkbutton(window,text='c',
                  variable=var2,onvalue=1,
                  offvalue=0,command=se
                  )
c1.pack()
c2.pack()


window.mainloop()