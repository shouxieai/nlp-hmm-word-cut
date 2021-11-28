all_data = open("all_train_text.txt","r",encoding = "utf-8").read().split("\n")[300:600]
with open("small_txt","w",encoding = "utf-8") as f:
    f.write("\n".join(all_data))