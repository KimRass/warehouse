# matplotlib
```python
import matplotlib as mpl
```
## mpl.font_manager.FontProperties().get_name()
```python
path = "C:/Windows/Fonts/malgun.ttf"
font_name = mpl.font_manager.FontProperties(fname=path).get_name()
```
## mpl.rc()
```python
mpl.rc("font", family=font_name)
```
```python
mpl.rc("axes", unicode_minus=False)
```
## matplotlib.pyplot
```python
import matplotlib.pyplot as plt
```
### plt.subplots()
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12)
```
### plt.setp()
```python
plt.setp(obj=ax1, yticks=ml_mean_gr_ax1["le"], yticklabels=ml_mean_gr_ax1.index)
```
### fig.colorbar()
```python
cbar = fig.colorbar(ax=ax, mappable=scatter)
```
### plt.style.use()
```python
plt.style.use("dark_background")
```
### plt.imshow()
```python
plt.imshow(image.numpy().reshape(3,3), cmap="Greys")
```
### fig.savefig()
```python
fig.savefig("means_plot_200803.png", bbox_inches="tight")
```
### fig.tight_layout()

### cbar.set_label()
```python
cbar.set_label(label="전용면적(m²)", size=15)
```
### ax.set()
```python
ax.set(title="Example", xlabel="xAxis", ylabel="yAxis", xlim=[0, 1], ylim=[-0.5, 2.5], xticks=data.index, yticks=[1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3])
```
### ax.set_title()
```python
ax.set_title("Example", size=20)
```
### ax.set_xlabel(), ax.set_ylabel()
```python
ax.set_xlabel("xAxis", size=15)
```
### ax.set_xlim(), ax.set_ylim()
```python
ax.set_xlim([1, 4])
```
### ax.axis()
```python
ax.axis([2, 3, 4, 10])
```
### ax.yaxis.set_tick_position()
```python
ax2.yaxis.set_ticks_position("right")
```
### ax.set_xticks(), ax.set_yticks()
```python
ax.set_xticks([1, 2])
```
### ax.tick_params()
```python
ax.tick_params(axis="x", labelsize=20, labelcolor="red", labelrotation=45, grid_linewidth=3)
```
### ax.legend()
```python
ax.legend(fontsize=14, loc="best")
```
### ax.grid()
```python
ax.grid(axis="x", color="White", alpha=0.3, linestyle="--", linewidth=2)
```
### ax.plot()
### ax.scatter()
```python
ax.scatter(x=gby["0.5km 내 교육기관 개수"], y=gby["실거래가"], s=70, c=gby["전용면적(m²)"], cmap="RdYlBu", alpha=0.7, edgecolors="black", linewidth=0.5)
```
### ax.barh()
```python
ax = plt.barh(y=ipark["index"], width=ipark["가경4단지 84.8743A"], height=0.2, alpha=0.5, color="red", label="가경4단지 84.8743A", edgecolor="black", linewidth=1)
```
### ax.hist()
```python
ax.hist(cnt_genre["genre"])
```
### ax.axhline()
```python
ax.axhline(y=mean, color="r", linestyle=":", linewidth=2)
```
### ax.text()
```python
ax1.text(y=row["le"]-0.2, x=row["abs_error"], s=round(row["abs_error"], 1), va="center", fontsize=12)
```
### plot(kind="pie")
```python
cnt_genre.sort_values("movie_id", ascending=False)["movie_id"].plot(ax=ax, kind="pie", startangle=90, legend=True)
```

# seaborn
```python
import seaborn as sb
```
### sb.scatterplot()
```python
ax = sb.scatterplot(data=df, x="ppa", y="error", hue="id", hue_norm=(20000, 20040), palette="RdYlGn", s=70, alpha=0.5)
```
### sb.lineplot()
```python
ax = sb.lineplot(x=data.index, y=data["ppa_ratio"], linewidth=3, color="red", label="흥덕구+서원구 아파트 평균")
ax = sb.lineplot(x=data.index, y=data["84A"], linewidth=2, color="green", label="가경아이파크 4단지 84A")
ax = sb.lineplot(x=data.index, y=data["84B"], linewidth=2, color="blue", label="가경아이파크 4단지 84B")
```
### sb.barplot()
```python
sb.barplot(ax=ax, x=area_df["ft_cut"], y=area_df[0], color="brown", edgecolor="black", orient="v")
```
### sb.replot()
```python
ax = sbrelplot(x="total_bill", y="tip", col="time", hue="day", style="day", kind="scatter", data=tips)
```
### sb.kedplot()
```python
ax = sb.kdeplot(np.array(data["ppa_root"]))
```
### sb.stripplot()
```python
ax = sb.stripplot(x=xx, y=yy, data=results_order, jitter=0.4, edgecolor="gray", size=4)
```
### sb.pairtplot()
```python
ax = sb.pairplot(data_for_corr)
```
### sb.heatmap()
* http://seaborn.pydata.org/generated/seaborn.heatmap.html
```python
corr = data.corr(method="spearman")
cmap = sb.diverging_palette(220, 10, as_cmap=True)
ax = sb.heatmap(corr, annot=True, annot_kws={"size": 10}, fmt=".2f", linewidths=0.2, cmap=cmap, center=0, vmin=-1, vmax=1)
```
