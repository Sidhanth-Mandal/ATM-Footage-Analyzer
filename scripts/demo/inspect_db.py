"""Print a short summary of the database (users, cases, job timings). Read-only."""
import sqlite3

c = sqlite3.connect("file:data/atmcctv.db?mode=ro", uri=True)
print("users:", list(c.execute("select id, username, role from user")))
print("cases:", list(c.execute('select id, ref_no, title from "case"')))
print("clips:", c.execute("select count(*) from clip").fetchone()[0])
for row in c.execute(
    "select type, status, count(*), round(avg(json_extract(result_json, '$.elapsed_s')), 1) from job group by type, status"
):
    print("job:", row)
