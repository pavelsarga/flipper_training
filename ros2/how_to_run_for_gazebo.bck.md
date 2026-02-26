omlouvam se za zpozdeni, narazil jsem na docela velky problem s celou tou codebase, protoze to nebylo moc navrzene na nejaky deployment na robota, tak jsem se snazil udelat co nejmin vybastleny reseni, abych v to mel aspon trochu duveru
https://github.com/edavidk7/flipper_training/blob/main/notebooks/ppo_policy_inference.ipynb
GitHub
flipper_training/notebooks/ppo_policy_inference.ipynb at main · ed...
Simulation engine and RL environment for training, optimization and control of VRAS tracked robots (marv, tradr) - edavidk7/flipper_training
flipper_training/notebooks/ppo_policy_inference.ipynb at main · ed...
tohle je muj repozitar, je to public, melo by byt v pohode to naklonovat
tady je jupyter notebook, ktery ukazuje, jak ten modul pouzit na vyextrahovani toho control commandu
Valentýn Číhala — 1/9/26, 11:14 AM
Snad se k tomu dostanu co nejdriv. Musim do pristiho patku sepsat minimum ale doufam ze o vikendu na to kouknu
Super diky
David K. — 1/9/26, 11:14 AM
jeste ti presne dodam uhlovou konvenci ktera je na flipperech a na tech vystupech 
snazil jsem se to napasovat na konvence rosu, tak snad to bude OK
jinak na tu heightmapu bych v provozu nastavil pomaly casovy decay (takovy to zapominani)
aby se moc neroztejkala
Valentýn Číhala — 1/9/26, 11:15 AM
ok
David K. — 1/9/26, 11:15 AM
tady je kdyztak primo ten soubor co pousti policy, kdyby bylo potreba do nej nejak sahat
https://github.com/edavidk7/flipper_training/blob/main/flipper_training/experiments/ppo/policy_inference_module.py
GitHub
flipper_training/flipper_training/experiments/ppo/policy_inference_...
Simulation engine and RL environment for training, optimization and control of VRAS tracked robots (marv, tradr) - edavidk7/flipper_training
Simulation engine and RL environment for training, optimization and control of VRAS tracked robots (marv, tradr) - edavidk7/flipper_training
je to jako pip-installable package, dodam ti tam tedka  requirementy aby se to dalo pyprojected nainstalovat se spravnyma verzema a otestuju, ze to bezi ve fresh venvu
David K. — 1/9/26, 12:19 PM
mas tam vsechno, co by melo stacit k rozbehnuti toho jupyteru lokalne
instaluju to pres uv teda
uv sync
David K. — 1/10/26, 4:55 PM
Ok, tak abych finalne doplnil ty konvence flipperu: zaporna rotacni rychlost (tzn. moje policy vrati pro ten flipper nejaky zaporny cislo) odpovida tomu, ze se predni flippery otaceji nahoru, zadni flippery se otaceji dolu. Pro tu policy je uhel 0 kazdeho flipperu kdyz je polozeny vodorovne. Takze kdyz je predni flipper uplne nahore, ma uhel -pi/2 (v moji codebase je to omezene, idealne to prosim na vstupu z ROSu clampni do tohohle intervalu) and kdyz je uplne dole tak ma pi/2. Pro zadni flipper je to obracene, tedy uplne dole ma -pi/2 a uplne nahore ma pi/2. Tohle ta policy ocekava.
Pro ostatni stavove veliciny, ten vektor ukazujici k cili musi byt v base link souradnem systemu v metrech, stejne tak vektor linearni rychlosti (v m/s), uhlove rychlosti (myslim ze tomu se v ROSu rika twist, v rad/s), pak uhly flipperu v radianech podle popsane konvence a jeste ROSacky quaternion (poradi x,y,z,w) vuci gravitacnimu vektoru (ja si z toho uvnitr vyberu roll/pitch). 
Pro tu heightmapu, chces ji mit ve formatu jako kdybys stal za robotem, dival se s nim dopredu a pak se predklonil nad nej -> veci pred robotem by mely byt v horni casti te heightmapy
Image /home/valda/workspaces/robot_rodeo_gym_ros2_ws/src/flipper_training/ros2/heightmap.png
policy ocekava rozliseni 64x64 s fyzickym rozsahem [1,1] (levy horni) az [-1,-1] (pravy dolni)
muzes tam dat vyssi rozliseni i vetsi rozmer, uvnitr si to resampluju na spravny
David K. — 1/10/26, 5:03 PM
tohle kdyztak doporucuju overit tim, ze si do toho nodu das na debugging plt.imshow(heightmap) a mel bys videt spravne pred/za robota
vystupni akce je: 4 rychlosti pasu v m/s (+1 dopredu, -1 dozadu), 4 rotacni rychlosti flipperu (znamenka jak jsem popsal nahore)
Valentýn Číhala — 1/11/26, 6:03 PM
ok jdu na to
David K. — 1/11/26, 6:03 PM
pis kdybys potreboval poradit
nebo volej