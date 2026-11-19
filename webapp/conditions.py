CONDITIONS = {
    "Acne": {
        "summary": "Blocked hair follicles that fill with oil and dead skin. Shows up as "
        "blackheads, whiteheads, red bumps or pus-filled spots, mostly on the face, "
        "chest and back.",
    },
    "Acute dermatitis, NOS": {
        "summary": "A recent, inflamed rash (red, itchy, sometimes weeping or crusting) "
        "that the dermatologists labelled as dermatitis without naming a more specific "
        "type. NOS stands for \"not otherwise specified\".",
    },
    "Allergic Contact Dermatitis": {
        "summary": "A delayed allergic reaction to something that touched the skin, such "
        "as nickel, fragrance, hair dye or poison ivy. An itchy red rash, sometimes with "
        "blisters, appears where the contact happened, usually a day or two later.",
    },
    "CD - Contact dermatitis": {
        "summary": "Contact dermatitis where the labels did not say whether the cause was "
        "an allergy or an irritant. The rash stays mostly on skin that touched the trigger.",
    },
    "Drug Rash": {
        "summary": "A skin reaction to a medicine. Most often a widespread rash of small "
        "red spots and bumps that starts days to a couple of weeks after a new drug.",
        "note": "Get medical help urgently if the rash blisters or peels, or comes with "
        "fever or sores in the mouth or eyes.",
    },
    "Eczema": {
        "summary": "Also called atopic dermatitis. Dry, itchy, inflamed skin that flares "
        "and settles over time, commonly in the creases of the elbows and knees, on the "
        "hands and on the face.",
    },
    "Folliculitis": {
        "summary": "Inflamed or infected hair follicles. Small red bumps or white-topped "
        "pimples, each centred on a hair, often after shaving, sweating or tight clothing.",
    },
    "Herpes Simplex": {
        "summary": "A recurring viral infection. Clusters of small, painful fluid-filled "
        "blisters on a red base, usually on or around the lips or genitals, that crust "
        "over within one to two weeks.",
    },
    "Herpes Zoster": {
        "summary": "Shingles: the chickenpox virus becoming active again. A painful band "
        "of blisters on one side of the body, often preceded by burning or tingling in "
        "the same area.",
        "note": "See a doctor soon. Antiviral treatment works best when started within "
        "about three days of the rash appearing, and a rash near the eye needs same-day care.",
    },
    "Impetigo": {
        "summary": "A contagious bacterial infection of the top layer of skin. Red sores "
        "that burst and leave honey-coloured crusts, often around the nose and mouth. "
        "Most common in children.",
        "note": "It spreads easily. A doctor can prescribe antibiotic cream or tablets.",
    },
    "Insect Bite": {
        "summary": "A reaction to an insect bite or sting. Itchy red bumps, sometimes "
        "with a tiny central puncture, often in groups or lines on exposed skin.",
    },
    "Irritant Contact Dermatitis": {
        "summary": "Skin damaged directly by something harsh, such as soap, detergent, "
        "solvents or frequent hand washing. Dry, red, cracked skin that burns or stings "
        "more than it itches.",
    },
    "Keratosis pilaris": {
        "summary": "A common, harmless condition. Small rough bumps, like goosebumps that "
        "don't go away, on the backs of the upper arms, thighs or cheeks.",
    },
    "Lichen Simplex Chronicus": {
        "summary": "Thick, leathery, often darker patches caused by scratching or rubbing "
        "the same spot over a long time. The itch and the scratching keep each other going.",
    },
    "Pigmented purpuric eruption": {
        "summary": "A group of harmless conditions where tiny blood vessels leak. "
        "Reddish-brown patches with pinpoint \"cayenne pepper\" spots, usually on the "
        "lower legs.",
    },
    "Pityriasis rosea": {
        "summary": "A rash that often starts with one larger oval \"herald patch\", "
        "followed a week or two later by smaller scaly oval patches on the trunk. "
        "It usually fades on its own within two to three months.",
    },
    "Psoriasis": {
        "summary": "A long-term immune condition. Well-defined, raised plaques covered "
        "with silvery scale, often on the elbows, knees, scalp and lower back.",
    },
    "Tinea": {
        "summary": "A fungal skin infection, also called ringworm. A ring-shaped scaly "
        "patch with a raised edge and a clearer centre. On the feet it is athlete's "
        "foot; in the groin, jock itch.",
    },
    "Tinea Versicolor": {
        "summary": "An overgrowth of a yeast that normally lives on the skin. Patches "
        "lighter or darker than the surrounding skin, with fine scale, mostly on the "
        "chest, back and shoulders.",
    },
    "Urticaria": {
        "summary": "Hives. Raised, itchy welts that appear and fade, each one usually "
        "lasting less than a day, though new ones can keep coming.",
        "note": "Get emergency help if hives come with swelling of the lips, tongue or "
        "throat, or with trouble breathing.",
    },
}


def describe(label):
    info = CONDITIONS.get(label, {})
    return {"summary": info.get("summary", ""), "note": info.get("note", "")}
