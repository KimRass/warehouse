- 출처 : https://medium.com/@andrejusb/machine-learning-date-feature-transformation-explained-4feb774c9dbe
## splitting features
- One of the ways is to split date value into multiple columns with numbers describing the original date (year, quarter, month, week, day of year, day of month, day of week).
## date feature transformation into a difference between dates
- We can use date difference as such:
— Day difference between Payment Due Date and Invoice Date
— Day difference between Payment Date and Invoice Date
This should bring clear pattern when there is payment delay — difference between payment date/invoice date will be bigger than between payment due date/invoice date.
