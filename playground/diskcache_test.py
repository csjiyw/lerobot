from pathlib import Path

from diskcache import Cache

# cache = Cache("/mnt/sdc/output/_ckpt")
# # task_id="abc"
# # # 写
# # cache[f"{task_id}_download"] = True
# cache.set("cached_task_id","7863799d68525d7e8fc7764ae0dce134")



n=Path("/root")/Path("/root/jyw_ws").relative_to("/")
print(n)
# res0,res1,res2,res3=None,None,None,None
# res0=cache.delete("abc")
# print(res0,res1,res2,res3,cache.get("abc"))
# res3=cache.set("abc",6)
# print(res0,res1,res2,res3,cache.get("abc"))
# res1=cache.add("abc",1)
# print(res0,res1,res2,res3,cache.get("abc"))
# res2=cache.add("abc",3)
# print(res0,res1,res2,res3,cache.get("abc"))
# res3=cache.set("abc",2)
# print(res0,res1,res2,res3,cache.get("abc"))

# # 读
# if cache.get(f"{task_id}_download"):
#     print("skip download")

# # 标记
# cache[f"{task_id}_cvt"] = True
# cache[f"{task_id}_cleanup"] = True