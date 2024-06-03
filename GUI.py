import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import ctypes
import numpy as np
import subprocess


class MovieSearchApp:
    def __init__(self, master):
        self.master = master
        master.title("电影推荐")

        # 获取屏幕分辨率
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        # 计算窗口左上角位置
        x = (screen_width - 800) / 2
        y = (screen_height - 360) / 2
        master.geometry(f"800x360+{int(x)}+{int(y)}")

        self.label = tk.Label(master, text="电影名:")
        self.label.grid(row=0, column=0, padx=10, pady=10)

        self.movie_name_entry = tk.Entry(master)
        self.movie_name_entry.grid(row=0, column=1, padx=10, pady=10)
        self.movie_name_entry.bind("<Return>", lambda event: self.search_movie())
        self.movie_name_entry.focus_set()

        self.genre_label = tk.Label(master, text="类型:")
        self.genre_label.grid(row=0, column=2, padx=10, pady=10)

        self.genre_combobox = ttk.Combobox(master, values=["全部"])
        self.genre_combobox.grid(row=0, column=3, padx=10, pady=10)
        self.genre_combobox.bind("<<ComboboxSelected>>", lambda event: self.search_movie())

        self.search_button = tk.Button(master, text="搜索", command=self.search_movie)
        self.search_button.grid(row=0, column=4, padx=10, pady=10)

        self.tree = ttk.Treeview(master, columns=("Year", "Country", "Genres", "Rating"), height=10)
        self.tree.heading("#0", text="电影名")
        self.tree.heading("Year", text="年代")
        self.tree.heading("Country", text="国家")
        self.tree.heading("Genres", text="类型")
        self.tree.heading("Rating", text="评分")
        self.tree.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=55, pady=10)

        self.tree.column("#0", width=200)
        self.tree.column("Year", width=80)
        self.tree.column("Country", width=100)
        self.tree.column("Genres", width=150)
        self.tree.column("Rating", width=150)

        self.tree.grid_rowconfigure(1, weight=1)

        self.genre_order = ["剧情", "喜剧", "动作", "爱情", "科幻", "动画", "悬疑", "惊悚", "恐怖", 
                            "纪录片", "短片", "情色", "音乐", "歌舞", "家庭", "儿童", "传记", "历史", 
                            "战争", "犯罪", "西部", "奇幻", "冒险", "灾难", "武侠", "古装", "运动", 
                            "同性", "戏曲", "真人秀", "脱口秀", "舞台艺术", "荒诞", "鬼怪"]
        
        self.load_movie_data()

        self.tree.bind("<Double-1>", self.rate_movie)  # 绑定双击事件

        self.generate_recommendation_button = tk.Button(master, text="生成推荐", command=self.generate_recommendation)
        self.generate_recommendation_button.grid(row=2, column=0, padx=10, pady=10)

        self.clear_data_button = tk.Button(master, text="清除评分记录", command=self.clear_data)
        self.clear_data_button.grid(row=2, column=4, padx=10, pady=10)

        self.search_movie()

    def load_movie_data(self):
        # 从CSV文件加载电影数据
        self.movies_df = pd.read_csv("./data/douban/movies.csv")
        # 将年代列转换为整数
        self.movies_df['YEAR'] = self.movies_df['YEAR'].astype(int)

        self.genre_combobox['values'] = ["全部"] + self.genre_order

    def format_rating(self, score, votes):
        if score == 0:
            return "暂无数据"
        else:
            return f"{score:.1f}/10, {int(votes)}人评价"

    def search_movie(self):
        # 清空之前的搜索结果
        for row in self.tree.get_children():
            self.tree.delete(row)

        # 获取搜索框中的电影名
        movie_name = self.movie_name_entry.get().strip().lower()
        selected_genre = self.genre_combobox.get()

        # 根据搜索条件匹配电影信息
        matched_movies = self.movies_df[self.movies_df['NAME'].str.lower().str.contains(movie_name, na=False)]
        if selected_genre != "全部":
            matched_movies = matched_movies[matched_movies['GENRES'].str.contains(selected_genre, na=False)]

        # 计算评分
        matched_movies['COMBINED_SCORE'] = matched_movies['DOUBAN_SCORE'] + np.log(matched_movies['DOUBAN_VOTES'] + 1)

        # 按评分排序
        matched_movies = matched_movies.sort_values(by=['COMBINED_SCORE'], ascending=False)

        # 显示在界面上
        for index, row in matched_movies.iterrows():
            rating_text = self.format_rating(row['DOUBAN_SCORE'], row['DOUBAN_VOTES'])
            self.tree.insert("", "end", text=row['NAME'], values=(row['YEAR'], row['REGIONS'], row['GENRES'], rating_text))

    def rate_movie(self, event):
        # 获取用户选中的电影名
        selected_item = self.tree.selection()[0]
        movie_name = self.tree.item(selected_item, "text")
        movie_id = self.movies_df.loc[self.movies_df['NAME'] == movie_name, 'MOVIE_ID'].iloc[0]  # 获取电影ID

        # 弹出评分界面
        rate_window = tk.Toplevel(self.master)
        rate_window.title(f"{movie_name}")

        # 获取屏幕分辨率
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)

        # 计算窗口左上角位置
        x = (screen_width - 250) / 2
        y = (screen_height - 100) / 2
        rate_window.geometry(f"250x100+{int(x)}+{int(y)}")

        # 创建评分标签和评分输入框
        rating_label = tk.Label(rate_window, text="评分(1-5):")
        rating_label.grid(row=0, column=0, padx=10, pady=10)
        rating_entry = tk.Entry(rate_window)
        rating_entry.grid(row=0, column=1, padx=10, pady=10)
        rating_entry.focus_set()

        # 创建提交按钮
        submit_button = tk.Button(rate_window, text="提交", command=lambda: self.submit_rating(rate_window, movie_id, rating_entry.get()))
        submit_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # 绑定回车键提交评分
        rate_window.bind("<Return>", lambda event: self.submit_rating(rate_window, movie_id, rating_entry.get()))

    def submit_rating(self, rate_window, movie_id, rating):
        try:
            rating = int(rating)
            if rating < 1 or rating > 5:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "评分为1-5的整数！")
        else:
            # 将评分数据记录到cold_start.csv文件中
            with open("./data/douban/cold_start.csv", "a") as file:
                # 检查是否需要换行
                if os.path.getsize("./data/douban/cold_start.csv") > 0:
                    file.write("\n")
                file.write(f"0,{movie_id},{rating}")
            messagebox.showinfo("成功", "评分已记录")
            rate_window.destroy()

    def generate_recommendation(self):
        if messagebox.askyesno("确认", "确认生成推荐？"):
            if self.check_cold_start_data():
                # messagebox.showinfo("提示", "正在生成推荐……")
                # 运行 predict.py 文件
                subprocess.run(["python", "predict.py"])
                messagebox.showinfo("提示", "请查看 output.md 文件。")
            else:
                messagebox.showinfo("提示", "没有评分数据，无法生成推荐！")

    def check_cold_start_data(self):
        return os.path.exists("./data/douban/cold_start.csv") and os.path.getsize("./data/douban/cold_start.csv") > 0

    def check_cold_start_directory(self):
        return os.path.exists("./data/douban/cold_start") and len(os.listdir("./data/douban/cold_start")) > 0

    def clear_data(self):
        if messagebox.askyesno("确认", "确认清除评分记录？"):
            try:
                # 清空 cold_start.csv 文件
                open("./data/douban/cold_start.csv", "w").close()
                # 清空 cold_start 目录
                if os.path.exists("./data/douban/cold_start"):
                    for file in os.listdir("./data/douban/cold_start"):
                        file_path = os.path.join("./data/douban/cold_start", file)
                        os.remove(file_path)
            except Exception as e:
                messagebox.showerror("错误", f"清除记录失败：{e}")
            else:
                messagebox.showinfo("成功", "记录已清除！")


root = tk.Tk()
app = MovieSearchApp(root)
root.mainloop()

