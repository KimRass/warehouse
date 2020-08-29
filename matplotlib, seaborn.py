# matplotlib
```
import matplotlib as mpl
```
## mpl.font_manager.FontProperties().get_name()
```
path = "C:/Windows/Fonts/malgun.ttf"
font_name = mpl.font_manager.FontProperties(fname=path).get_name()
```
## mpl.rc()
```
mpl.rc("font", family=font_name)
```
```
mpl.rc("axes", unicode_minus=False)
```
## matplotlib.pyplot
```
import matplotlib.pyplot as plt
```
### plt.subplots()
```
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12)
```
### plt.setp()
```
plt.setp(obj=ax1, yticks=ml_mean_gr_ax1["le"], yticklabels=ml_mean_gr_ax1.index)
```
### fig.colorbar()
```
cbar = fig.colorbar(ax=ax, mappable=scatter)
```
### plt.style.use()
```
plt.style.use("dark_background")
```
### plt.imshow()
```
plt.imshow(image.numpy().reshape(3,3), cmap="Greys")
```
### fig.savefig()
```
fig.savefig("D:/☆디지털혁신팀/☆실거래가 분석/☆그래프/means_plot_200803.png", bbox_inches="tight")
```
### fig.tight_layout()

### cbar.set_label()
```
cbar.set_label(label="전용면적(m²)", size=15)
```
### ax.set()
```
ax.set(title="Example", xlabel="xAxis", ylabel="yAxis", xlim=[0, 1], ylim=[-0.5, 2.5], xticks=data.index, yticks=[1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
```
### ax.set_title()
```
ax.set_title("Example", size=20)
```
### ax.set_xlabel(), ax.set_ylabel()
```
ax.set_xlabel("xAxis", size=15)
```
### ax.set_xlim(), ax.set_ylim()
```
ax.set_xlim([1, 4])
```
### ax.axis()
```
ax.axis([2, 3, 4, 10])
```
### ax.yaxis.set_tick_position()
```
ax2.yaxis.set_ticks_position("right")
```
### ax.set_xticks(), ax.set_yticks()
```
ax.set_xticks([1, 2])
```
### ax.tick_params()
```
ax.tick_params(axis="x", labelsize=20, labelcolor="red", labelrotation=45, grid_linewidth=3)
```
### ax.legend()
```
ax.legend(fontsize=14, loc="best")
```
### ax.grid()
```
ax.grid(axis="x", color="White", alpha=0.3, linestyle="--", linewidth=2)
```
### ax.plot()
### ax.scatter()
```
ax.scatter(x=gby["0.5km 내 교육기관 개수"], y=gby["실거래가"], s=70, c=gby["전용면적(m²)"], cmap="RdYlBu", alpha=0.7, edgecolors="black", linewidth=0.5)
```
### ax.barh()
```
ax = plt.barh(y=ipark["index"], width=ipark["가경4단지 84.8743A"], height=0.2, alpha=0.5, color="red", label="가경4단지 84.8743A", edgecolor="black", linewidth=1)
```
### ax.hist()
```
ax.hist(cnt_genre["genre"])
```
### ax.axhline()
```
ax.axhline(y=mean, color="r", linestyle=":", linewidth=2)
```
### ax.text()
```
ax1.text(y=row["le"]-0.2, x=row["abs_error"], s=round(row["abs_error"], 1), va="center", fontsize=12)
```
### plot(kind="pie")
```
cnt_genre.sort_values("movie_id", ascending=False)["movie_id"].plot(ax=ax, kind="pie", startangle=90, legend=True)
```

# seaborn
```
import seaborn as sb
```
### sb.scatterplot()
```
ax = sb.scatterplot(data=df, x="ppa", y="error", hue="id", hue_norm=(20000, 20040), palette="RdYlGn", s=70, alpha=0.5)
```
### sb.lineplot()
```
ax = sb.lineplot(x=data.index, y=data["ppa_ratio"], linewidth=3, color="red", label="흥덕구+서원구 아파트 평균")
ax=sb.lineplot(x=data.index, y=data["84A"], linewidth=2, color="green", label="가경아이파크 4단지 84A")
ax=sb.lineplot(x=data.index, y=data["84B"], linewidth=2, color="blue", label="가경아이파크 4단지 84B")
```
### sb.barplot()
```
sb.barplot(ax=ax, x=area_df["ft_cut"], y=area_df[0], color="brown", edgecolor="black", orient="v")
```
### sb.replot()
```
ax = sbrelplot(x="total_bill", y="tip", col="time", hue="day", style="day", kind="scatter", data=tips)
```
### sb.kedplot()
```
ax = sb.kdeplot(np.array(data["ppa_root"]))
```
### sb.stripplot()
```
ax = sb.stripplot(x=xx, y=yy, data=results_order, jitter=0.4, edgecolor="gray", size=4)
```
### sb.pairtplot()
```
ax = sb.pairplot(data_for_corr)
```
### sb.heatmap()
* 출처 : [http://seaborn.pydata.org/generated/seaborn.heatmap.html](http://seaborn.pydata.org/generated/seaborn.heatmap.html) 
```
corr = data.corr(method="spearman")
cmap = sb.diverging_palette(220, 10, as_cmap=True)
ax = sb.heatmap(corr, annot=True, annot_kws={"size": 10}, fmt=".2f", linewidths=0.2, cmap=cmap, center=0, vmin=-1, vmax=1)
```