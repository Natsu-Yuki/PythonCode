import tkinter as tk
window=tk.Tk()
window.title('Natsu')
window.geometry('300x200')

e=tk.Entry(window,show=None)
e.pack()

def insert_point():
    var=e.get()
    t.insert('insert',var)


def insert_end():
    var=e.get()
    t.insert('end',var)



b1=tk.Button(
    window,text='insert point',width=15,height=2,
    command=insert_point
            )

b1.pack()



b2=tk.Button(
    window,text='insert end',width=15,height=2,
    command=insert_end
            )

b2.pack()

t=tk.Text(window,height=5)
t.pack()

window.mainloop()
