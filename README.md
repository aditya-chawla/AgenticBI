# AgenticBI

## AdventureWorks Test Queries
Here are 10 end-to-end natural language queries you can use to test the Agentic BI platform with the AdventureWorks dataset. These test various SQL capabilities (aggregations, joins, time-series, filtering) and trigger different types of visualizations on the dashboard:

1. **"Show me the top 10 best-selling products by total revenue."**
   - *Tests:* Aggregation (`SUM`), Joining, Grouping, Ordering (`DESC`), Limit.
   - *Expected Visualization:* Bar Chart.

2. **"What is the total sales revenue by year and month?"**
   - *Tests:* Date parsing and extraction, grouping by time periods, aggregation.
   - *Expected Visualization:* Line Chart.

3. **"Show me the distribution of employees by marital status."**
   - *Tests:* Simple grouping and `COUNT()` aggregation.
   - *Expected Visualization:* Pie/Donut chart.

4. **"What is the average sick leave hours by job title?"**
   - *Tests:* Categorical grouping using an `AVG()` function.
   - *Expected Visualization:* Horizontal Bar Chart (due to dense categories).

5. **"Show me total sales amount by sales territory country."**
   - *Tests:* Joining sales and territory tables, geospatial/categorical context.
   - *Expected Visualization:* Bar chart (or Map).

6. **"Which top 5 vendors do we spend the most money with?"**
   - *Tests:* Purchasing schema joins (`PurchaseOrderHeader` and `Vendor`).
   - *Expected Visualization:* Bar chart.

7. **"Show me the total inventory quantity grouped by product subcategory."**
   - *Tests:* multi-table joins (`ProductInventory`, `Product`, `ProductSubcategory`).
   - *Expected Visualization:* Horizontal Bar chart.

8. **"Compare the standard cost versus the list price for all products."**
   - *Tests:* Retrieving continuous numeric variables without aggregation.
   - *Expected Visualization:* Scatter Plot.

9. **"Show me the number of orders placed online versus in-store."**
   - *Tests:* Using boolean flags (`OnlineOrderFlag`) to group and count.
   - *Expected Visualization:* Pie chart or Bar chart.

10. **"Who are the top 5 sales representatives by total sales, and what were their total sales?"**
    - *Tests:* Complex conditional joins spanning Sales and Person schemas.
    - *Expected Visualization:* Bar chart ranking the reps.