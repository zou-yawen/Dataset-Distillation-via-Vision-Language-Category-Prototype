from collections import OrderedDict


tiny_imagenet_CLASSES = OrderedDict({
    "n01443537": "goldfish, Carassius auratus",
    "n01629819": "European fire salamander, Salamandra salamandra",
    "n01641577": "bullfrog, Rana catesbeiana",
    "n01644900": "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "n01698640": "American alligator, Alligator mississipiensis",
    "n01742172": "boa constrictor, Constrictor constrictor",
    "n01768244": "trilobite",
    "n01770393": "scorpion",
    "n01774384": "black widow, Latrodectus mactans",
    "n01774750": "tarantula",
    "n01784675": "centipede",
    "n01855672": "goose",
    "n01882714": "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "n01910747": "jellyfish",
    "n01917289": "brain coral",
    "n01944390": "snail",
    "n01945685": "slug",
    "n01950731": "sea slug, nudibranch",
    "n01983481": "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "n01984695": "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "n02002724": "black stork, Ciconia nigra",
    "n02056570": "king penguin, Aptenodytes patagonica",
    "n02058221": "albatross, mollymawk",
    "n02074367": "dugong, Dugong dugon",
    "n02085620": "Chihuahua",
    "n02094433": "Yorkshire terrier",
    "n02099601": "golden retriever",
    "n02099712": "Labrador retriever",
    "n02106662": "German shepherd, German shepherd dog, German police dog, alsatian",
    "n02113799": "standard poodle",
    "n02123045": "tabby, tabby cat",
    "n02123394": "Persian cat",
    "n02124075": "Egyptian cat",
    "n02125311": "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "n02129165": "lion, king of beasts, Panthera leo",
    "n02132136": "brown bear, bruin, Ursus arctos",
    "n02165456": "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "n02190166": "fly",
    "n02206856": "bee",
    "n02226429": "grasshopper, hopper",
    "n02231487": "walking stick, walkingstick, stick insect",
    "n02233338": "cockroach, roach",
    "n02236044": "mantis, mantid",
    "n02268443": "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "n02279972": "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "n02281406": "sulphur butterfly, sulfur butterfly",
    "n02321529": "sea cucumber, holothurian",
    "n02364673": "guinea pig, Cavia cobaya",
    "n02395406": "hog, pig, grunter, squealer, Sus scrofa",
    "n02403003": "ox",
    "n02410509": "bison",
    "n02415577": "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "n02423022": "gazelle",
    "n02437312": "Arabian camel, dromedary, Camelus dromedarius",
    "n02480495": "orangutan, orang, orangutang, Pongo pygmaeus",
    "n02481823": "chimpanzee, chimp, Pan troglodytes",
    "n02486410": "baboon",
    "n02504458": "African elephant, Loxodonta africana",
    "n02509815": "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "n02666196": "abacus",
    "n02669723": "academic gown, academic robe, judge's robe",
    "n02699494": "altar",
    "n02730930": "apron",
    "n02769748": "backpack, back pack, knapsack, packsack, rucksack, haversack",
    "n02788148": "bannister, banister, balustrade, balusters, handrail",
    "n02791270": "barbershop",
    "n02793495": "barn",
    "n02795169": "barrel, cask",
    "n02802426": "basketball",
    "n02808440": "bathtub, bathing tub, bath, tub",
    "n02814533": "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    "n02814860": "beacon, lighthouse, beacon light, pharos",
    "n02815834": "beaker",
    "n02823428": "beer bottle",
    "n02837789": "bikini, two-piece",
    "n02841315": "binoculars, field glasses, opera glasses",
    "n02843684": "birdhouse",
    "n02883205": "bow tie, bow-tie, bowtie",
    "n02892201": "brass, memorial tablet, plaque",
    "n02906734": "broom",
    "n02909870": "bucket, pail",
    "n02917067": "bullet train, bullet",
    "n02927161": "butcher shop, meat market",
    "n02948072": "candle, taper, wax light",
    "n02950826": "cannon",
    "n02963159": "cardigan",
    "n02977058": "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    "n02988304": "CD player",
    "n02999410": "chain",
    "n03014705": "chest",
    "n03026506": "Christmas stocking",
    "n03042490": "cliff dwelling",
    "n03085013": "computer keyboard, keypad",
    "n03089624": "confectionery, confectionary, candy store",
    "n03100240": "convertible",
    "n03126707": "crane",
    "n03160309": "dam, dike, dyke",
    "n03179701": "desk",
    "n03201208": "dining table, board",
    "n03250847": "drumstick",
    "n03255030": "dumbbell",
    "n03355925": "flagpole, flagstaff",
    "n03388043": "fountain",
    "n03393912": "freight car",
    "n03400231": "frying pan, frypan, skillet",
    "n03404251": "fur coat",
    "n03424325": "gasmask, respirator, gas helmet",
    "n03444034": "go-kart",
    "n03447447": "gondola",
    "n03544143": "hourglass",
    "n03584254": "iPod",
    "n03599486": "jinrikisha, ricksha, rickshaw",
    "n03617480": "kimono",
    "n03637318": "lampshade, lamp shade",
    "n03649909": "lawn mower, mower",
    "n03662601": "lifeboat",
    "n03670208": "limousine, limo",
    "n03706229": "magnetic compass",
    "n03733131": "maypole",
    "n03763968": "military uniform",
    "n03770439": "miniskirt, mini",
    "n03796401": "moving van",
    "n03804744": "nail",
    "n03814639": "neck brace",
    "n03837869": "obelisk",
    "n03838899": "oboe, hautboy, hautbois",
    "n03854065": "organ, pipe organ",
    "n03891332": "parking meter",
    "n03902125": "pay-phone, pay-station",
    "n03930313": "picket fence, paling",
    "n03937543": "pill bottle",
    "n03970156": "plunger, plumber's helper",
    "n03976657": "pole",
    "n03977966": "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    "n03980874": "poncho",
    "n03983396": "pop bottle, soda bottle",
    "n03992509": "potter's wheel",
    "n04008634": "projectile, missile",
    "n04023962": "punching bag, punch bag, punching ball, punchball",
    "n04067472": "reel",
    "n04070727": "refrigerator, icebox",
    "n04074963": "remote control, remote",
    "n04099969": "rocking chair, rocker",
    "n04118538": "rugby ball",
    "n04133789": "sandal",
    "n04146614": "school bus",
    "n04149813": "scoreboard",
    "n04179913": "sewing machine",
    "n04251144": "snorkel",
    "n04254777": "sock",
    "n04259630": "sombrero",
    "n04265275": "space heater",
    "n04275548": "spider web, spider's web",
    "n04285008": "sports car, sport car",
    "n04311004": "steel arch bridge",
    "n04328186": "stopwatch, stop watch",
    "n04356056": "sunglasses, dark glasses, shades",
    "n04366367": "suspension bridge",
    "n04371430": "swimming trunks, bathing trunks",
    "n04376876": "syringe",
    "n04398044": "teapot",
    "n04399382": "teddy, teddy bear",
    "n04417672": "thatch, thatched roof",
    "n04456115": "torch",
    "n04465501": "tractor",
    "n04486054": "triumphal arch",
    "n04487081": "trolleybus, trolley coach, trackless trolley",
    "n04501370": "turnstile",
    "n04507155": "umbrella",
    "n04532106": "vestment",
    "n04532670": "viaduct",
    "n04540053": "volleyball",
    "n04560804": "water jug",
    "n04562935": "water tower",
    "n04596742": "wok",
    "n04597913": "wooden spoon",
    "n06596364": "comic book",
    "n07579787": "plate",
    "n07583066": "guacamole",
    "n07614500": "ice cream, icecream",
    "n07615774": "ice lolly, lolly, lollipop, popsicle",
    "n07695742": "pretzel",
    "n07711569": "mashed potato",
    "n07715103": "cauliflower",
    "n07720875": "bell pepper",
    "n07734744": "mushroom",
    "n07747607": "orange",
    "n07749582": "lemon",
    "n07753592": "banana",
    "n07768694": "pomegranate",
    "n07871810": "meat loaf, meatloaf",
    "n07873807": "pizza, pizza pie",
    "n07875152": "potpie",
    "n07920052": "espresso",
    "n09193705": "alp",
    "n09246464": "cliff, drop, drop-off",
    "n09256479": "coral reef",
    "n09332890": "lakeside, lakeshore",
    "n09428293": "seashore, coast, seacoast, sea-coast",
    "n12267677": "acorn"
}
)