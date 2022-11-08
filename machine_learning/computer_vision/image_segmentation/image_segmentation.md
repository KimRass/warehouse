# Image Segmentation
- 각 픽셀마다 Class별 확률 값을 출력합니다. 따라서 Output shape은 `(height, width, n_class)`가 됩니다.

# Types of Image Segmentation
- Reference: https://keymakr.com/blog/instance-vs-semantic-segmentation/#:~:text=In%20other%20words%2C%20semantic%20segmentation,a%20dataset%20for%20instance%20segmentation.
- Semantic Segmentation
  - Semantic segmentation. Objects shown in an image are grouped based on defined categories. For instance, a street scene would be segmented by "pedestrians," "bikes," "vehicles," "sidewalks," and so on.
- Instance segmentation
  - Consider instance segmentation a refined version of semantic segmentation. Categories like "vehicles" are split into "cars," "motorcycles," "buses," and so on — instance segmentation detects the instances of each category.
- In other words, semantic segmentation treats multiple objects within a single category as one entity. Instance segmentation, on the other hand, identifies individual objects within these categories.